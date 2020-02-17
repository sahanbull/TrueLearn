from collections import defaultdict

import numpy as np
import trueskill

from truelearn_experiments.utils import get_summary_stats, get_topic_dict


def v_func(x):
    return (pdf(x) / cdf(x))


def w_func(x):
    v = v_func(x)
    return (v * (v + x))


def pdf(x, mu=0, sigma=1):
    """Probability density function"""
    return (1 / np.sqrt(2 * np.pi) * abs(sigma) *
            np.exp(-(((x - mu) / abs(sigma)) ** 2 / 2)))


def cdf(x, mu=0, sigma=1):
    """Cumulative distribution function"""
    return 0.5 * erfc(-(x - mu) / (sigma * np.sqrt(2)))


def erfc(x):
    """Complementary error function (via `http://bit.ly/zOLqbc`_)"""
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
            0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
            0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277
    )))
    )))
    )))
    return 2. - r if x < 0 else r


def get_default_variance_from_coverage_values(data, type="cosine"):
    # get variance of all topic coverage values in the dataset
    if type == "binary":
        return 1.0

    topic_coverage = np.var(data.
                            values().
                            flatMap(lambda events: [coverage for event in events
                                                    for coverage in
                                                    get_topic_dict(event["topics"], type=type).values()]).
                            collect())

    return topic_coverage


def compute_c(beta_sqr, skill_var):
    return np.sqrt(beta_sqr + skill_var)


def _erfc(x):
    """Complementary error function (via `http://bit.ly/zOLqbc`_)"""
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
            0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
            0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277
    )))
    )))
    )))
    return 2. - r if x < 0 else r


def _cdf(x, mu=0, sigma=1):
    """Cumulative distribution function"""
    return 0.5 * erfc(-(x - mu) / (sigma * np.sqrt(2)))


def team_sum_quality(mean_skill_user, var_skill_user, mean_skill_content, var_skill_content, beta):
    """Algorithm to compute the quality using the difference of means in a cumulative manner

    Args:
        mean_skill_user ([float): list of means of skills of the learner
        var_skill_user ([float]): list of variances of skills of the learner
        mean_skill_content ([float]): list of means of topics of the content
        var_skill_content ([float]): list of variances of topics of the content
        beta (float): beta value (should be the standard deviation, not the variance)

    Returns:
        (float): probability of mean difference
    """

    difference = np.sum(mean_skill_user) - np.sum(mean_skill_content)
    std = np.sqrt(np.sum(np.sqrt(var_skill_user)) + np.sum(np.sqrt(var_skill_content)) + beta)
    return _cdf(difference, 0, std)


def fixed_depth_trueskill_model(records, init_skill=0., def_var=None, tau=0., beta_sqr=0., threshold=0.5,
                                engage_func=None, positive_only=True):
    """This model calculates trueskill given all positive skill using the real trueskill factorgraph.
    Args:
        records [[val]]: list of vectors for each event of the user. Format of vector
            [session, time, timeframe_id, topic_id, topic_cov ..., label]

    Returns:
        accuracy (float): accuracy for all observations
        concordance ([bool]): the concordance between actual and predicted values
    """
    # setup trueskill environment
    trueskill.setup(mu=0.0, sigma=1 / 1000000000, beta=float(np.sqrt(beta_sqr)), tau=tau, draw_probability=0.,
                    backend="mpmath")

    num_records = float(len(records))

    if num_records <= 1:
        return 0., [], int(num_records), False

    user_model = {
        "mean": {},
        "variance": {}
    }

    topics_covered = set()

    actual = []
    predicted = []

    stats = defaultdict(int)

    prev_label = None

    for idx, event in enumerate(records):
        #  calculate if the user is going to engage with this resource
        topic_vec = event[1:-1]
        topic_dict = get_topic_dict(topic_vec)

        # track unique topic encountered
        topics_covered |= set(topic_dict.keys())

        # create_teams
        team_learner = tuple()
        team_learner_mean_vec = list()
        team_learner_var_vec = list()

        team_content = tuple()
        team_content_mean_vec = list()
        team_content_var_vec = list()

        topic_seq = []

        for topic, coverage in topic_dict.items():
            topic_seq.append(topic)
            # get user skill rating
            tmp_learner_mean = user_model["mean"].get(topic, init_skill)
            tmp_learner_var = user_model["variance"].get(topic, def_var)
            learner_skill = trueskill.Rating(mu=float(tmp_learner_mean),
                                             sigma=float(np.sqrt(tmp_learner_var)))

            team_learner += (learner_skill,)
            team_learner_mean_vec.append(tmp_learner_mean)
            team_learner_var_vec.append(tmp_learner_var)

            # get skill coverage
            tmp_coverage = coverage
            tmp_content_var = float(np.square(1 / 1000000000))
            topic_cov = trueskill.Rating(mu=tmp_coverage, sigma=float(np.sqrt(tmp_content_var)))
            team_content += (topic_cov,)
            team_content_mean_vec.append(tmp_coverage)
            team_content_var_vec.append(tmp_content_var)

        # check if user engages
        pred_prob = team_sum_quality(np.array(team_learner_mean_vec),
                                     np.array(team_learner_var_vec),
                                     np.array(team_content_mean_vec),
                                     np.array(team_content_var_vec), np.sqrt(beta_sqr))

        prediction = int(pred_prob >= threshold)

        # if user engages, update the model
        label = event[-1]

        # if label is negative and setting is positive only, skip updating
        if positive_only and label != 1:
            pass
        else:
            # if positive
            if label == 1:
                # learner wins
                new_team_learner, _ = trueskill.rate([team_learner, team_content], ranks=[0, 1])
            else:
                # content wins
                _, new_team_learner = trueskill.rate([team_content, team_learner], ranks=[0, 1])

            for _idx, topic in enumerate(topic_seq):
                user_model["mean"][topic], user_model["variance"][
                    topic] = new_team_learner[_idx].mu, new_team_learner[_idx].sigma ** 2

        # if not first element, calculate accuracy
        if idx != 0:
            if label != prev_label:
                stats["change_label"] += 1

            actual.append(label)
            predicted.append(prediction)

        prev_label = label

    stats = dict(stats)

    stats["num_topics"] = len(topics_covered)

    accuracy, precision, recall, f1, stats = get_summary_stats(actual, predicted, num_records, stats=stats,
                                                               user_model=user_model)

    return accuracy, precision, recall, f1, int(num_records), stats


def convert_mappings_to_vectors(user_model, topic_dict, def_var):
    # create the vectors for the new model
    topic_seq = []
    u_mean = []
    u_var = []
    t_cov = []

    # dict to array
    for topic, coverage in topic_dict.items():
        topic_seq.append(topic)
        u_mean.append(user_model["mean"].get(topic, 0))
        u_var.append(user_model["variance"].get(topic, def_var))
        t_cov.append(coverage)

    return topic_seq, np.array(u_mean), np.array(u_var), np.array(t_cov)
