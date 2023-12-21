import argparse
import numpy as np
import json
from lr_4_task_9 import pearson_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Recommendation system')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser


# Generate recommendations for the input user
def generate_recommendations(dataset, user):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    total_scores = {}
    similarity_sums = {}

    for u in [x for x in dataset if x != user]:
        similarity_score = pearson_score(dataset, user, u)

        if similarity_score <= 0:
            continue

        filetered_list = [x for x in dataset[u] if x not in dataset[user] or dataset[user][x] == 0]

        for item in filetered_list:
            total_scores.update({item: dataset[u][item] * similarity_score})
            similarity_sums.update({item: similarity_score})

    if len(total_scores) == 0:
        return ['No recommendations possible']

    # Create the normalized list
    movie_ranks = np.array([[total / similarity_sums[item], item] for item, total in total_scores.items()])

    # Sort in decreasing order based on the first column
    movie_ranks = movie_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

    # Extract the recommended movies
    recommendations = [movie for _, movie in movie_ranks]

    return recommendations


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    raitings_file = 'lab4/data/ratings.json'
    with open(raitings_file, 'r') as f:
        data = json.loads(f.read())

    print("\nRecommendations for " + user + ":")
    movies = generate_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i + 1) + '. ' + movie)