import itertools
from collections import defaultdict

import numpy as np

from truelearn_experiments.utils import get_topic_dict, get_summary_stats


def fit_backg_bernouilli(user_model, engagement, pguess=0.25, pfail=0.):
    """

    Args:
        user_model ({int: float}): topic id, prob of knowing of each topic of the user
        engagement (bool):
        pguess (float):
        pfail (float):

    Returns:

    """
    # user_model is a vector representing the probability of the user knowing each topic
    # topics is a vector representing the binary coverage of the content
    # bool_eng is a boolean related to engagement.
    # pguess is the probability of not having the appropriate knowledge but still being engaged
    # pfail is the probability of having the appropriate knowledge but not being engaged

    # IMPORTANT: 1) with this model we would assume that it is equally likely that the learner
    # knows or doesn't know a topic a priori. This is, we need to initialise skills to 0.5, not to 0.
    # 2) The model assumes that once someone learns something completely (prob = 1) you can't unlearn
    # it. So if a skill is 1 it won't move.

    # get skills used
    relevant_skills = list(user_model.keys())
    relevant_skills_probs = list(user_model.values())
    k = len(relevant_skills_probs)

    # get table of binary combinations for bernouilli variables
    truthTable = list(itertools.product([0, 1], repeat=k))
    truthTable = np.array(truthTable)

    # number of binary combinations
    ncomb = len(truthTable)
    # vector that keeps the result of computing AND operator for the binary combinations
    and_vector = np.zeros(ncomb)

    # Incorporate pguess and pfail to bool_eng
    is_engaged_prob = engagement * (1 - pguess)

    # compute AND
    for i in range(ncomb):
        and_vector[i] = np.prod(truthTable[i])

    table_probs = np.zeros((ncomb, k + 1))
    # compute table of probabilities for binary skills
    for i in range(k):
        for j in range(ncomb):
            # ipdb.set_trace()
            table_probs[j, i] = truthTable[j, i] * relevant_skills_probs[i] + (not (truthTable[j, i])) * (
                    1 - relevant_skills_probs[i])

    # compute table of probabilities for AND
    for j in range(ncomb):
        table_probs[j, k] = and_vector[j] * is_engaged_prob + (not (and_vector[j])) * (1 - is_engaged_prob)

    new_skill = np.zeros(k)
    # for each variable we compute the update
    for i in range(k):
        ptrue = 0
        pfalse = 0
        for j in range(ncomb):
            # the update is computed as a product of all other messages (except the one we are considering) summing up over all variables
            table_probs_tmp = np.delete(table_probs[j], i)
            product = np.prod(table_probs_tmp)
            ptrue = ptrue + (truthTable[j, i] * product)
            pfalse = pfalse + ((not (truthTable[j, i])) * product)

        # compute update
        update = ptrue / (ptrue + pfalse)
        # multiply by prior
        new_skill[i] = (relevant_skills_probs[i] * update) / (
                relevant_skills_probs[i] * update + (1 - relevant_skills_probs[i]) * (1 - update))

    new_user_model = {k: new_skill[idx] for idx, k in enumerate(relevant_skills)}

    return new_user_model


def predict_backg_bernouilli(user_model, pguess=0.25, pfail=0.0, threshold=0.5):
    """

    Args:
        user_model:
        topic_dict:
        pguess:
        pfail:
        threshold:

    Returns:
        bool: True if prob > threshold, else False
    """

    # get skills used
    relevant_skills = list(user_model.keys())
    relevant_skills_probs = list(user_model.values())
    k = len(relevant_skills_probs)

    # get table of binary combinations for bernouilli variables
    truthTable = list(itertools.product([0, 1], repeat=k))
    truthTable = np.array(truthTable)

    # number of binary combinations
    ncomb = len(truthTable)
    # vector that keeps the result of computing AND operator for the binary combinations
    and_vector = np.zeros(ncomb)

    # compute AND
    for i in range(ncomb):
        and_vector[i] = np.prod(truthTable[i])

    table_probs = np.zeros((ncomb, k))
    # compute table of probabilities for binary skills
    for i in range(k):
        for j in range(ncomb):
            # ipdb.set_trace()
            table_probs[j, i] = truthTable[j, i] * relevant_skills_probs[i] + (not (truthTable[j, i])) * (
                    1 - relevant_skills_probs[i])

    ptrue = 0
    pfalse = 0
    for j in range(ncomb):
        product = np.prod(table_probs[j])
        ptrue = ptrue + (and_vector[j] * product)
        pfalse = pfalse + ((not (and_vector[j])) * product)

        # compute prob
        prob_tmp = ptrue / (ptrue + pfalse)

    # Incorporate pguess and pfail to bool_eng
    prob = prob_tmp * (1 - pguess)

    # ipdb.set_trace()
    return prob


def multi_skill_kt_model(records, def_var=0.5, tau=0., beta_sqr=0., engage_func="all", threshold=0.5,
                         positive_only=True):
    """This model calculates trueskill given all positive skill.
    Args:
        records [[val]]: list of vectors for each event of the user. Format of vector
            [session, time, timeframe_id, topic_id, topic_cov ..., label]
        def_var (float): initial uncertainty
        tau (float): p_guess
        beta_sqr (float): p_fail
        engage_fun: function that estimates engagement probability
        threshold (float): engagement threshold

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

    actual = []
    predicted = []

    stats = defaultdict(int)

    prev_label = None

    for idx, event in enumerate(records):
        #  calculate if the user is going to engage with this resource
        topics = event[1:-1]
        topic_dict = get_topic_dict(topics)

        # track unique topic encountered
        topics_covered |= set(topic_dict.keys())

        # check if user engages
        temp_user_model = {k: user_model["mean"].get(k, def_var) for k in topic_dict}

        prediction = int(
            predict_backg_bernouilli(temp_user_model, threshold=threshold, pguess=tau, pfail=beta_sqr)) >= threshold

        # if user engages, update the model
        label = event[-1]

        # if label is negative and setting is positive only, skip updating
        if positive_only and label != 1:
            pass
        else:
            # update if label is positive or negative
            temp_label = bool(label)
            temp_user_model = fit_backg_bernouilli(temp_user_model, temp_label, pguess=tau, pfail=beta_sqr)

            for topic, p_know in temp_user_model.items():
                user_model["mean"][topic] = p_know

        # if not first element, calculate accuracy
        if idx != 0:
            if label != prev_label:
                stats["change_label"] += 1

            actual.append(label)
            predicted.append(prediction)

        prev_label = label

    stats = dict(stats)

    # num topics
    stats["num_topics"] = len(topics_covered)

    accuracy, precision, recall, f1, stats = get_summary_stats(actual, predicted, num_records, stats=stats,
                                                               user_model=user_model)

    stats["user_model"] = None

    return accuracy, precision, recall, f1, int(num_records), stats
