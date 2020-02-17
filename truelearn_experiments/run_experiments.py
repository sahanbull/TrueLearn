from functools import partial
from operator import itemgetter
import numpy as np

from pyspark.sql import SparkSession

from truelearn_experiments.naive_baseline_models import persistent_model, majority_model, engage_model

import json

from truelearn_experiments.fixed_depth_trueskill_models import get_default_variance_from_coverage_values, \
    fixed_depth_trueskill_model, truelearn_model

from truelearn_experiments.multi_skill_kt_models import multi_skill_kt_model
from truelearn_experiments.utils import convert_to_records


def _get_topic_vector(topics, type):
    if type == "cosine":
        return topics
    else:
        n_topics = int(len(topics) / 2)
        new_topics = []

        if type == "binary":
            for i in range(n_topics):
                topic_idx = i * 2
                new_topics += [topics[topic_idx], True]

        return new_topics


def vectorise_data(events, vector_type):
    events.sort(key=itemgetter("time", "timeframe"))

    return [
        [l["session"]] + _get_topic_vector(l["topics"], vector_type) + [l["label"]] for l in events]


def _get_eval_func(algorithm, vect_type, data=None, engage_func="all", threshold=.5, def_var_factor=0.5, tau_factor=0.1,
                   beta_factor=.1, draw_probability="static", positive_only=True, draw_factor=.1):
    if algorithm == "engage":
        return engage_model

    elif algorithm == "persistent":
        return persistent_model

    elif algorithm == "majority":
        return majority_model

    elif algorithm == "multi_skill_kt":
        _def_var = def_var_factor  # .5
        _beta_sqr = beta_factor  # pfail
        _tau = tau_factor  # pguess
        return partial(multi_skill_kt_model, def_var=_def_var, tau=_tau, beta_sqr=_beta_sqr,
                       engage_func="all",
                       threshold=threshold, positive_only=positive_only)

    elif algorithm == "fixed_depth_trueskill":
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
        _tau = float(1. * tau_factor)
        return partial(fixed_depth_trueskill_model, init_skill=0., def_var=_def_var, tau=_tau,
                       beta_sqr=_beta_sqr,
                       engage_func=engage_func, threshold=threshold, positive_only=positive_only)

    elif algorithm == "truelearn":
        cosine_var = float(get_default_variance_from_coverage_values(data, vect_type))
        _def_var = float(np.square(cosine_var * def_var_factor))
        _beta_sqr = float(np.square(np.sqrt(_def_var) * beta_factor))
        _tau = float(1. * tau_factor)
        return partial(truelearn_model, init_skill=0., def_var=_def_var, tau=_tau, beta_sqr=_beta_sqr,
                       engage_func=engage_func, threshold=threshold, draw_probability=draw_probability,
                       positive_only=positive_only, draw_factor=draw_factor)


def restructure_data(line):
    (sess, (acc, prec, rec, f1, num_events, is_stats)) = line

    temp_dict = {
        "session": sess,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "num_events": num_events
    }

    if is_stats:
        temp_dict["positive_rate"] = is_stats["positive"]
        temp_dict["predict_positive_rate"] = is_stats["predict_positive"]
        temp_dict["change_label_rate"] = is_stats["change_label"]
        temp_dict["num_topics"] = is_stats["num_topics"]
        temp_dict["num_topics_rate"] = is_stats["num_topics_rate"]
        temp_dict["num_user_topics"] = is_stats["num_user_topics"]

        temp_dict["user_model"] = is_stats["user_model"]

    return temp_dict


def main(args):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load training data
    data = (spark.read.csv(args["dataset_filepath"], sep=",", header=False).
            rdd.
            map(convert_to_records))

    grouped_data = data.map(lambda l: (l["session"], l)).groupByKey(numPartitions=8).mapValues(list)

    # run the algorithm to get results
    if (args["algorithm"]) == "fixed_depth_trueskill":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   engage_func=args["engage_func"], def_var_factor=args["def_var_factor"],
                                   tau_factor=args["tau_factor"], beta_factor=args["beta_factor"],
                                   threshold=args["threshold"], positive_only=args["positive_only"])

    elif (args["algorithm"]) == "truelearn":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], data=grouped_data,
                                   engage_func=args["engage_func"], def_var_factor=args["def_var_factor"],
                                   tau_factor=args["tau_factor"], beta_factor=args["beta_factor"],
                                   threshold=args["threshold"], draw_probability=args["draw_probability"],
                                   draw_factor=args["draw_factor"], positive_only=args["positive_only"])

    elif (args["algorithm"]) == "multi_skill_kt":
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"], def_var_factor=args["def_var_factor"],
                                   tau_factor=args["tau_factor"], beta_factor=args["beta_factor"],
                                   threshold=args["threshold"], positive_only=args["positive_only"])


    else:
        eval_func = _get_eval_func(args["algorithm"], args["skill_repr"])

    evaluated_data = grouped_data.mapValues(lambda l: vectorise_data(l, args["skill_repr"])).mapValues(eval_func)

    restructured_data = evaluated_data.map(restructure_data).collect()

    with open(args["output_dir"] + "{}_model_results.json".format(args["algorithm"]), "w") as outfile:
        json.dump(restructured_data, outfile)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--algorithm', default='truelearn', const='all', nargs='?',
                        choices=['engage', 'persistent', 'majority', "truelearn", "fixed_depth_trueskill",
                                 "multi_skill_kt"],
                        help="The name of the algorithm can be one of the allowed algorithms")
    parser.add_argument('--skill-repr', default='cosine', const='all', nargs='?',
                        choices=['cosine', 'binary'],
                        help="How the skills should be represented in the models")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    parser.add_argument('--engage-func', default='all', const='all', nargs='?',
                        choices=["all"],
                        help="What engagement eval function to be used")
    parser.add_argument('--threshold', type=float, default=.5,
                        help="Probability threshold for classifying true")
    parser.add_argument('--def-var-factor', type=float, default=1000.,
                        help="Probability of knowing this topics")
    parser.add_argument('--tau-factor', type=float, default=.0,
                        help="Probability of watching even when cant learn")
    parser.add_argument('--beta-factor', type=float, default=.5,
                        help="Probability skipping even when can learn")
    parser.add_argument('--draw-probability', type=str, default="individual",
                        help="Probability of drawing the match")
    parser.add_argument('--draw-factor', type=float, default=.03,
                        help="factor of draw probability to be used")
    parser.add_argument('--positive-only', action='store_true', help="learns from negative examples too")

    args = vars(parser.parse_args())

    main(args)
