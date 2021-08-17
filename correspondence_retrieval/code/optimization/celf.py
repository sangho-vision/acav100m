import time

from tqdm import tqdm


def celf(
    measure,
    dataset_size,
    subset_size,
    start_indices,
    intermediate_target=None,
    clustering_combinations=None
):
    candidates = list(set(range(dataset_size)) - set(start_indices))
    # random.shuffle(candidates)

    start_time = time.time()
    marg_gain = [
        measure(start_indices + [ind], clustering_combinations=clustering_combinations)
        for ind in candidates
    ]

    Q = sorted(
        zip(candidates, marg_gain),
        key=lambda x: x[1][0],
        reverse=True,
    )

    agreed_dict = Q[0][1][1]
    Q = [(fst, snd[0]) for fst, snd in Q]

    S = start_indices + [Q[0][0]]
    gain = Q[0][1]
    GAIN = [Q[0][1]]

    Q, LOOKUPS, timelapse = (
        Q[1:],
        [len(candidates)],
        [time.time() - start_time]
    )

    pbar = tqdm(range(len(start_indices), subset_size - 1), desc='celf iter')
    for _ in pbar:
        check, lookup = False, 0

        while not check:
            lookup += 1

            current = Q[0][0]

            current_gain, current_agreed_dict = measure(
                S + [current], clustering_combinations=clustering_combinations,
                agreed_dict=agreed_dict,
            )
            Q[0] = (current, current_gain - gain)

            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            check = (Q[0][0] == current)

        agreed_dict = current_agreed_dict
        gain += Q[0][1]
        S.append(Q[0][0])
        GAIN.append(gain)
        LOOKUPS.append(lookup)
        timelapse.append(time.time() - start_time)

        if intermediate_target is not None:
            precision = len(set(intermediate_target) & set(S)) / len(set(S))
            pbar.set_description("(LEN: {}, MEASURE: {}, PRECISION: {})".format(
                len(S), gain, precision))
        else:
            pbar.set_description("(LEN: {}, MEASURE: {})".format(len(S), gain))

        Q = Q[1:]

    return (S, GAIN, timelapse, LOOKUPS)
