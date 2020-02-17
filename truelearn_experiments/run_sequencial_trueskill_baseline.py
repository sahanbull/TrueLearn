from collections import defaultdict
from os.path import join

import trueskill
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from truelearn_experiments.fixed_depth_trueskill_models import team_sum_quality
from truelearn_experiments.utils import convert_to_records, get_topic_dict


def get_accuracy_values(user_actual, user_predicted, user_counts, user_unique_topic_count):
    users = list(set(user_actual.keys()))
    users.sort()

    weights = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    metric_records = []

    for user in users:
        actual = user_actual[user]
        predicted = user_predicted[user]

        num_unique_topics = int(user_unique_topic_count[user])
        num_events = int(user_counts[user])

        weights.append(num_events)

        tmp_accuracy_score = accuracy_score(actual, predicted, normalize=True)
        tmp_precision_score = precision_score(actual, predicted)
        tmp_recall_score = recall_score(actual, predicted)
        tmp_f1_score = f1_score(actual, predicted)

        accuracies.append(tmp_accuracy_score)
        precisions.append(tmp_precision_score)
        recalls.append(tmp_recall_score)
        f1s.append(tmp_f1_score)

        metric_records.append({
            "session": user,
            "trueskill_accuracy": tmp_accuracy_score,
            "trueskill_precision": tmp_precision_score,
            "trueskill_recall": tmp_recall_score,
            "trueskill_f1": tmp_f1_score,
            "num_topics_rate": num_unique_topics / float(num_events)
        })

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    metrics_df = pd.DataFrame([{
        "trueskill_accuracy": avg_accuracy,
        "trueskill_precision": avg_precision,
        "trueskill_recall": avg_recall,
        "trueskill_f1": avg_f1
    }])

    avg_accuracy_w = np.average(accuracies, weights=weights)
    avg_precision_w = np.average(precisions, weights=weights)
    avg_recall_w = np.average(recalls, weights=weights)
    avg_f1_w = np.average(f1s, weights=weights)

    metrics_w_df = pd.DataFrame([{
        "trueskill_accuracy": avg_accuracy_w,
        "trueskill_precision": avg_precision_w,
        "trueskill_recall": avg_recall_w,
        "trueskill_f1": avg_f1_w
    }])

    metrics_df = metrics_df[["trueskill_accuracy", "trueskill_precision", "trueskill_recall", "trueskill_f1"]]
    metrics_w_df = metrics_w_df[["trueskill_accuracy", "trueskill_precision", "trueskill_recall", "trueskill_f1"]]

    return pd.DataFrame(metric_records), metrics_df, metrics_w_df


def main(args):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load training data
    data = (spark.read.csv(args["dataset_filepath"], sep=",", header=False).
            rdd.
            map(convert_to_records))

    timewise_data = data.collect()
    timewise_data.sort(key=lambda l: (l["time"], l["timeframe"]))

    def_mu = float(25.)
    def_sigma = float(def_mu / 3.)
    def_beta = float(def_sigma / 2.)

    # initiate trueskill
    trueskill.setup(mu=def_mu, sigma=def_sigma, beta=def_beta, tau=0., draw_probability=0., backend="mpmath")

    # initiate performance recorders
    user_event_count = defaultdict(int)
    user_unique_topic_count = defaultdict(set)
    user_predicted = defaultdict(list)
    user_actual = defaultdict(list)

    # initiate learner models
    user_models = {}

    # initiate video, topic params
    engageability_models = {}

    all_topics = set()

    for event in timewise_data:
        slug = event["slug"]
        vid_id = event["vid_id"]
        user = event["session"]

        topics = get_topic_dict(event["topics"]).keys()

        all_topics |= set(topics)

        user_unique_topic_count[user] |= set(topics)

        user_model = user_models.get(user, {"mean": {},
                                            "variance": {}
                                            })

        slug_vid_id = (slug, vid_id)
        topic_model = engageability_models.get(slug_vid_id, {"mean": {},
                                                             "variance": {}
                                                             })

        # create_teams
        team_learner = tuple()
        team_learner_mean_vec = list()
        team_learner_var_vec = list()

        team_content = tuple()
        team_content_mean_vec = list()
        team_content_var_vec = list()

        topic_seq = []

        for topic in topics:
            topic_seq.append(topic)
            # get user skill rating
            tmp_learner_mean = user_model["mean"].get(topic, def_mu)
            tmp_learner_var = float(user_model["variance"].get(topic, np.square(def_sigma)))
            learner_skill = trueskill.Rating(mu=float(tmp_learner_mean),
                                             sigma=float(np.sqrt(tmp_learner_var)))

            team_learner += (learner_skill,)
            team_learner_mean_vec.append(tmp_learner_mean)
            team_learner_var_vec.append(tmp_learner_var)

            # get skill coverage
            tmp_coverage = topic_model["mean"].get(topic, def_mu)
            tmp_content_var = float(topic_model["variance"].get(topic, np.square(def_sigma)))
            topic_cov = trueskill.Rating(mu=tmp_coverage, sigma=float(np.sqrt(tmp_content_var)))
            team_content += (topic_cov,)
            team_content_mean_vec.append(tmp_coverage)
            team_content_var_vec.append(tmp_content_var)

        # check if user engages
        pred_prob = team_sum_quality(np.array(team_learner_mean_vec),
                                     np.array(team_learner_var_vec),
                                     np.array(team_content_mean_vec),
                                     np.array(team_content_var_vec), def_beta)

        prediction = int(pred_prob >= .5)

        label = event["label"]

        # update
        if label == 1:
            # learner wins
            new_team_learner, new_team_content = trueskill.rate([team_learner, team_content], ranks=[0, 1])
        else:
            # content wins
            new_team_content, new_team_learner = trueskill.rate([team_content, team_learner], ranks=[0, 1])

        # recreate user and content model representations
        for _idx, topic in enumerate(topic_seq):
            user_model["mean"][topic], user_model["variance"][
                topic] = new_team_learner[_idx].mu, new_team_learner[_idx].sigma ** 2

            topic_model["mean"][topic], topic_model["variance"][
                topic] = new_team_learner[_idx].mu, new_team_learner[_idx].sigma ** 2

        # assign it to the new dictionary
        user_models[user] = user_model
        engageability_models[slug_vid_id] = topic_model

        # if not first element, calculate accuracy
        user_event_count[user] += 1
        if user_event_count[user] != 1:
            user_actual[user].append(label)
            user_predicted[user].append(prediction)

    user_unique_topic_count = {user_id: len(unique_topics) for user_id, unique_topics in
                               user_unique_topic_count.items()}

    accuracy_per_user, accuracy, weighted_accuracy = get_accuracy_values(user_actual, user_predicted, user_event_count,
                                                                         user_unique_topic_count)

    # print()
    accuracy_per_user.to_csv(join(args["output_dir"], "summary_results.csv"), index=False)
    accuracy.to_csv(join(args["output_dir"], "summary_accuracy.csv"), index=False)
    weighted_accuracy.to_csv(join(args["output_dir"], "summary_accuracy_weighted.csv"), index=False)


if __name__ == '__main__':
    """this script takes in the wikified lectures file and the learner activity data from videolectures to build a .
    output of this script will be {slug, vid_id, part_id, start_time, stop_time, clean, text, wiki_concepts}
    eg: command to run this script:

    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-filepath', type=str, required=True,
                        help="where training data is")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory path where the results will be saved.")
    args = vars(parser.parse_args())

    main(args)
