from collections import defaultdict

import trueskill

from truelearn_experiments.utils import get_topic_dict, get_summary_stats
import numpy as np


def truelearn_model(records, init_skill=0., def_var=None, tau=0., beta_sqr=0., threshold=0.5, engage_func=None,
                    draw_probability="static", draw_factor=.1, positive_only=True):
    """This model calculates trueskill given all positive skill using the real trueskill factorgraph.
    Args:
        records [[val]]: list of vectors for each event of the user. Format of vector
            [session, time, timeframe_id, topic_id, topic_cov ..., label]

    Returns:
        accuracy (float): accuracy for all observations
        concordance ([bool]): the concordance between actual and predicted values
    """

    num_records = float(len(records))

    if num_records <= 1:
        return 0., [], int(num_records), False

    user_model = {
        "mean": {},
        "variance": {}
    }

    topics_covered = set()

    actual = [1 / 1000000000]  # as the draw probability cant be zero
    predicted = []

    stats = defaultdict(int)

    prev_label = None

    for idx, event in enumerate(records):

        #  calculate if the user is going to engage with this resource
        topic_vec = event[1:-1]
        topic_dict = get_topic_dict(topic_vec)

        # setup trueskill environment
        if draw_probability == "static":
            # _draw_probability = float(0.5932538086581619)  # population success rate
            _draw_probability = 1.
        else:
            # compute novelty prob
            _draw_probability = float(np.mean(actual))
            # _draw_probability = float(novel_prob)  # individual.. majority model

        _draw_probability *= draw_factor
        _draw_probability = 1 - 1 / 1000000000 if _draw_probability == 1. else _draw_probability

        trueskill.setup(mu=0.0, sigma=1 / 1000000000, beta=float(np.sqrt(beta_sqr)), tau=tau,
                        draw_probability=_draw_probability,
                        backend="mpmath")

        # track unique topic encountered
        topics_covered |= set(topic_dict.keys())

        # create_teams
        team_learner = tuple()
        team_mean_learner = []

        team_content = tuple()
        team_mean_content = []

        topic_seq = []

        for topic, coverage in topic_dict.items():
            topic_seq.append(topic)
            # get user skill rating
            tmp_learner_skill = user_model["mean"].get(topic, init_skill)
            learner_skill = trueskill.Rating(mu=tmp_learner_skill,
                                             sigma=np.sqrt(user_model["variance"].get(topic, def_var)))

            team_learner += (learner_skill,)
            team_mean_learner.append(tmp_learner_skill)

            # get skill coverage
            tmp_content_topic = coverage
            topic_cov = trueskill.Rating(mu=tmp_content_topic, sigma=1 / 1000000000)
            team_content += (topic_cov,)
            team_mean_content.append(tmp_content_topic)

        # check if user engages
        pred_prob = trueskill.quality([team_learner, team_content])
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
                new_team_learner, _ = trueskill.rate([team_learner, team_content], ranks=[0, 0])
            else:  # if the person is not engaged...
                # check if the winner is learner or content
                difference = np.sum(team_mean_learner) - np.sum(team_mean_content)

                if difference > 0.:  # learner wins
                    new_team_learner, _ = trueskill.rate([team_learner, team_content], ranks=[0, 1])
                elif difference < 0.:  # learner loses
                    _, new_team_learner = trueskill.rate([team_content, team_learner], ranks=[0, 1])
                else:
                    new_team_learner = team_learner

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

    accuracy, precision, recall, f1, stats = get_summary_stats(actual[1:], predicted, num_records, stats=stats,
                                                               user_model=user_model)

    return accuracy, precision, recall, f1, int(num_records), stats
