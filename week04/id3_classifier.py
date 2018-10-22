import numpy as np


class ID3Classifier:

    def __init__(self, target_column, most_common):
        self.tree_node_list = {}
        self.next_node = None
        self.target_column_name = target_column
        self.most_common = most_common

    def calculate_branch_entropy(self, targets):
        """determine the entropy for the branch so the info gain can be determined"""
        branch_entropy = 0
        total_targets = len(targets)
        class_frequencies = []
        unique_classes = np.unique(targets)
        for c in unique_classes:
            class_frequencies.append(targets.count(c) / total_targets)
        for probability in class_frequencies:
            branch_entropy -= self.calculate_entropy(probability)
        return branch_entropy

    def calculate_entropy(self, probability):
        if probability != 0:
            return probability * np.log2(probability)
        else:
            return 0

    def determine_feature_entropy(self, targets, feature_values, feature_indexes):
        feature_entropy = 0
        for value in feature_values:
            branch_targets = []
            for index in feature_indexes[value]:
                # print("feature {} class: {}".format(value, targets[index]))
                branch_targets.append(targets[index])
            feature_entropy += self.calculate_branch_entropy(branch_targets) * (len(branch_targets) / len(targets))

        # print(feature_entropy)
        return feature_entropy

    def create_tree(self, data, targets, column_names):
        num_rows = len(data)
        num_features = len(column_names)
        most_common_class = max(set(targets), key=targets.count)

        if num_rows == 0 or num_features == 0:
            return most_common_class
        elif targets.count(targets[0]) == num_rows:
            return targets[0]

        else:
            feature_entropies = []
            all_feature_values = []
            for feature in column_names:
                feature_values = []
                feature_indexes = {}
                # this holds the indexes for which data values follow down the branch
                filtered_data = []
                index = 0
                for value in data[feature]:
                    if value not in feature_values:
                        feature_values.append(value)
                        feature_indexes[value] = [index]
                        index += 1
                    else:
                        feature_indexes[value].append(index)
                        index += 1
                # print(feature_indexes)
                all_feature_values.append(feature_values)
                feature_entropies.append(self.determine_feature_entropy(targets, feature_values, feature_indexes))
            best_feature = feature_entropies.index(min(feature_entropies))
            best_feature_name = column_names[best_feature]
            tree_node_list = {best_feature_name: {}}
            new_names = [x for i, x in enumerate(column_names) if i != best_feature]
            # print(new_names)
            # print(tree_node_list)

            # now check each branch of this feature
            for value in all_feature_values[best_feature]:
                new_data = data.loc[data[best_feature_name] == value]
                # print(new_data)
                subtree = self.create_tree(new_data, new_data[self.target_column_name].values.tolist(), new_names)

                tree_node_list[best_feature_name][value] = subtree
            return tree_node_list

    def fit(self, dataframe, targets, column_names):
        tree_model = TreeModel(column_names, self.most_common)
        tree_model.tree_node_list = self.create_tree(dataframe, targets, column_names)
        return tree_model


class TreeModel:

    def __init__(self, column_names, most_common):
        self.tree_node_list = None
        self.node_names = column_names
        self.most_common = most_common

    def predict(self, test_data):
        predicted_targets = []

        for column_name in test_data:
            if column_name in self.tree_node_list:
                # print("first node is: " + column_name)
                for idx, row in test_data.iterrows():
                    # print(self.tree_node_list[column_name])
                    predicted_targets.append(self.traverse_tree(self.tree_node_list, row, column_name))

        return predicted_targets

    def traverse_tree(self, dictionary, data, branch_name):

        # in the event that there is no pre-existing branch for this route, create a leaf node with most frequent class
        dictionary[branch_name] = dictionary.get(branch_name, self.most_common)

        if not isinstance(dictionary[branch_name], dict):
            return dictionary[branch_name]

        if branch_name not in self.node_names:
            # print(list(dictionary[branch_name].keys())[0])
            target_value = self.traverse_tree(dictionary[branch_name], data, list(dictionary[branch_name].keys())[0])
        else:
            # print(data[branch_name])
            target_value = self.traverse_tree(dictionary[branch_name], data, data[branch_name])

        return target_value
