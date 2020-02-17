from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _get_user_model(user_model):
    _user_model = {}
    # get the topics
    topics = set(user_model["mean"].keys()) | set(user_model["variance"].keys())
    for topic in topics:
        _user_model[topic] = (user_model["mean"][topic], user_model["variance"].get(topic))

    return _user_model


def get_summary_stats(actual, predicted, num_records, stats=None, user_model=None):
    """

    Args:
        actual:
        predicted:
        stats:
        user_model:

    Returns:

    """
    # compute classification metrics
    accuracy = accuracy_score(actual, predicted, normalize=True)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    if stats is not None:
        num_topics = float(stats["num_topics"])

        # get user model
        stats["user_model"] = _get_user_model(user_model)

        # get topic rate
        stats["num_topics_rate"] = num_topics / num_records

        #  get change label rate
        stats["change_label"] = stats.get("change_label", 0.) / num_records

        # in user model
        stats["num_user_topics"] = len(stats["user_model"].keys())

        # get positive rate
        stats["positive"] = len([i for i in actual if i == 1]) / num_records
        stats["predict_positive"] = len([i for i in predicted if i == 1]) / num_records

    return accuracy, precision, recall, f1, stats


def convert_to_records(record):
    """convert csv of learner events to event records

    Args:
        record: (slug, vid_id, time, timeframe, session, --topics--, label)

    Returns:

    """
    record = list(record)
    return {
        "slug": str(record[0]),
        "vid_id": int(float(record[1])),
        "time": float(record[2]),
        "timeframe": int(float(record[3])),
        "session": str(record[4]),
        "topics": [float(i) for i in record[5:-1]],
        "label": int(float(record[-1]))
    }


def get_topic_dict(topics, type="cosine"):
    """

    Args:
        topics [float]: a list of numbers where the even indices are the topic ids and odd indices are coverage values
        type (str): type of repr, can be cosine, norm or binary
    Returns:
        {int: float}: a dict with topic id: topic coverage
    """
    num_topics = int(len(topics) / 2)
    topic_dict = {}
    covs = []
    for i in range(num_topics):
        topic_id_idx = i * 2
        topic_cov_idx = topic_id_idx + 1

        topic_id = int(topics[topic_id_idx])
        topic_cov = float(topics[topic_cov_idx])

        topic_dict[topic_id] = topic_cov
        covs.append(topic_cov)

    if type == "cosine":
        return topic_dict
    elif type == "binary":
        return {topic: True for topic in topic_dict}
    else:  # norm transformation
        norm = float(sum(covs))
        return {topic: cov / norm for topic, cov in topic_dict.items()}
