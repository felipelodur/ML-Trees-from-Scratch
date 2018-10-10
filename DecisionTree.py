from pprint import pprint
import pandas as pd
import numpy as np
import random

class DecisionTree(object):
    """
    """
    def __init__(self, max_depth = 3, min_samples = 2, impurity = 'entropy'):
        self.max_depth = max_depth
        self.min_samples = min_samples
        assert impurity in ['entropy','gini'],"Invalid impurity method, choose 'entropy' or 'gini'"
        self._impurity = impurity
    
    # TREINAMENTO.....
    def _check_purity(self, data):
        """ helper """
        label_column = data[:, -1]
        qty_unique_classes = len(np.unique(label_column))
    
        return not (qty_unique_classes - 1)
    
    def _classify_data(self, data):
        """ helper """
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        prevalent_class_index = counts_unique_classes.argmax()
        classification = unique_classes[prevalent_class_index]

        return unique_classes, counts_unique_classes
    
    def _get_potential_splits(self, data):
        """ helper """
        potential_splits = {}
        n_columns = data.shape[1]
        
        rolling_mean = lambda a: ((a[1:] + a[:-1])/2).tolist()

        for column_index in range(n_columns - 1):        # excluding the last column which is the label
            values = data[:, column_index]
            unique_values = np.unique(values)

            potential_splits[column_index] = rolling_mean(unique_values)

        return potential_splits
    
    def _split_data(self, data, split_column, split_value):
        """ helper """
        split_column_values = data[:, split_column]
        
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]

        return data_below, data_above
    
    def _calculate_impurity(self, data):
        """ helper """
        label_column = data[:, -1]
        counts = np.unique(label_column, return_counts=True)[1] # note que np.unique so retorna labels encontrados...

        probabilities = counts / counts.sum()
        if self._impurity == 'gini':
            impurity = - probabilities @ np.log2(probabilities) # ... assim não vai existir 0 * inf, o que no lim daria 0 mesmo
        else: # entropy otherwise
            impurity = probabilities @ (1-probabilities)
            
        return impurity
    
    
    def _calculate_overall_impurity(self, data_below, data_above):
        """ helper """
        lens_ba = np.array([len(data_below), len(data_above)])
        probs_ba = lens_ba / lens_ba.sum()

        entropy_ba = np.array([self._calculate_impurity(data_below), 
                               self._calculate_impurity(data_above)])

        overall_entropy =  probs_ba @ entropy_ba
        
        return overall_entropy
    
    def _determine_best_split(self, data, potential_splits):
        """ helper """
        overall_entropy = 9999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self._split_data(data, 
                                                          split_column=column_index, 
                                                          split_value=value)
                current_overall_entropy = self._calculate_overall_impurity(data_below, 
                                                                          data_above)
                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value

        return best_split_column, best_split_value
    
    def _stop_split(self, data, counter):
        """
        helper. O nó deve parar de ser dividido?
        incluir todas as condições para parar de dividir o nó
        returns
        -------
        stop_split:
            True: se o nó for terminal, i.e., uma das condições de parada foi atingido
            False: se o nó for intermediário, pode ir seguir quebrando
        """
        stop_split = self._check_purity(data) or (len(data) < self.min_samples) or \
                    (counter == self.max_depth)
        return stop_split
    
    def fit(self, df):
        """
        função externa para montar a arvore
        """
        self.tree_ = self._decision_tree_algorithm(df)
    
    def _decision_tree_algorithm(self, df, counter = 0):
        """
        procedimento recursivo para montar a arvore
        """
    
        # data preparations
        if counter == 0:
            #global COLUMN_HEADERS
            self.column_headers = df.columns
            data = df.values
        else:
            data = df           

        # base cases
        basecase_reached = self._stop_split(data, counter)
        if basecase_reached:
            classification, counts_unique_classes = self._classify_data(data)
            return classification, counts_unique_classes

        # recursive part
        else:    
            counter += 1

            # helper functions 
            potential_splits = self._get_potential_splits(data)
            split_column, split_value = self._determine_best_split(data, potential_splits)
            data_below, data_above = self._split_data(data, split_column, split_value)

            # instantiate sub-tree
            feature_name = self.column_headers[split_column]
            question = "{} <= {}".format(feature_name, split_value)
            sub_tree = {question: []}

            # find answers (recursion)
            yes_answer = self._decision_tree_algorithm(data_below, counter)
            no_answer  = self._decision_tree_algorithm(data_above, counter)

            # If the answers are the same, then there is no point in asking the qestion.
            # This could happen when the data is classified even though it is not pure
            # yet (min_samples or max_depth base case).
            if np.array_equal(yes_answer, no_answer):
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree
    
    # PREDICOES.....
    def predict(self, df):
        """
        função externa para predição. 
        percorre a arvore em busca de um nó terminal
        """
        prediction = df.apply(self._classify_example, axis = 1)
        prediction = prediction.apply(pd.Series)
        prediction.columns = ['pred', 'proba']
        
        return prediction
    
    def _calc_probs(self, a):
        """ helper, retorna vetor normalizado por L1"""
        norm = a[1].sum()
        res = {k: v/norm for k, v in zip(a[0], a[1])}
        return res

    def _get_most_prevalent_class(self, a):
        """ helper, retorna a classe mais prevalente"""
        most_prevalent_idx = a[1].argmax()
        most_prevalent_class = a[0][most_prevalent_idx]
        return most_prevalent_class
    
    def _classify_example(self, example, tree = None):
        """ proc recursivo que retorna a classificação e as probabilidades
        """
        
        if not tree:  # prim. chamada, usar arvore inteira
            tree = self.tree_
        
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split(" ")

        # ask question
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        # base case
        if not isinstance(answer, dict):
            probs = self._calc_probs(answer)
            the_class = self._get_most_prevalent_class(answer)
            return the_class, probs

        # recursive part
        else:
            residual_tree = answer
            return self._classify_example(example, residual_tree)
    