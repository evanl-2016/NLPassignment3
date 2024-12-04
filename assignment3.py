##For HMM
import numpy as np
import conllu
import utils
import io
import random
import collections
import math
from scipy.special import logsumexp
from scipy.cluster.vq import whiten, kmeans, vq, kmeans2
##For BERT
import torch
from transformers import BertTokenizer, BertModel

###
###  Subtask 1: HMM POS tagger using unsupervised learning using the Baum-Welch algorithm
###

class HMMPOSTagger:

    def __init__(self, conlluFile, label, unknown_threshold, convergence_threshold, test, max_iter):
        # Open file ready for recording performance data
        eval_file = open("HMMeval.txt","w")
        # Emission probs as an array of dictionaries and transition probabilites in a matrix
        self.emmission_probs = [{}]
        self.transition_matrix = []
        # Get states and vocab
        (states,vocab, sentence_count) = self.get_states_and_vocab(conlluFile, label, unknown_threshold)
        states = np.append(states, np.array(["<START>", "<END>"]))
        #vocab = np.array(list(set(untagged_data)))
        vocab = np.append(vocab, np.array(["<start>","<end>"]))
        # Initialise emission and transmission probs
        for s in range(len(states)):
            self.emmission_probs.append({})
            for v in vocab:
                self.emmission_probs[s][v] = random.random()
            sum = np.sum(list(self.emmission_probs[s].values()))
            self.emmission_probs[s] = {v: np.log(self.emmission_probs[s][v]/sum) for v in vocab}
        for i in range(len(states)):
            self.transition_matrix.append([])
            for j in range(len(states)):
                self.transition_matrix[i].append(random.random())
            sum = np.sum(self.transition_matrix[i])
            self.transition_matrix[i] = [np.log(self.transition_matrix[i][j]/sum) for j in range(len(states))]
        # Parse data into numpy array with form and POS as fields
        previous_transition_matrix = self.transition_matrix
        previous_emmission_matrix = self.emmission_probs
        curr_iter = 0
        while True:
            curr_iter += 1
            dataFile = open(conlluFile, "r")
            self.generator = conllu.parse_incr(dataFile)
            supervised_data = []
            current_sentence = 0
            for sentence in self.generator:
                supervised_data = []
                for token in sentence:
                    supervised_data.append((token["form"], token["xpos"]) if token["form"] in vocab else ("<unknown>", "UNKNOWN"))
                supervised_data = np.array(supervised_data,dtype=[("form", "U20"), ("POS","U8")])
                untagged_data = supervised_data["form"]
                # Initialise transition and observation
                untagged_data = np.append(untagged_data, "<end>")
                untagged_data = np.append(["<start>"], untagged_data)
                # Run the forward-backward algorithm
                self.forward_backward(untagged_data, states, vocab)
                current_sentence += 1
                print("Progress: " + str((current_sentence / sentence_count) * 100) + "%")
                if current_sentence/sentence_count > 0.25 and test:
                    break
            # Generate evaluation stats and store
            eval_data_file = open(conlluFile, "r")
            eval_generator = conllu.parse_incr(eval_data_file)
            supervised_data = []
            current_sentence = 0
            accuracies = []
            precisions = []
            recalls = []
            homo_scores = []
            comp_scores = []
            v_scores = []
            for sentence in eval_generator:
                current_sentence += 1
                supervised_data = []
                tags = []
                for token in sentence:
                    supervised_data.append(token["form"] if token["form"] in vocab else "<unknown>")
                    tags.append(token[label] if token["form"] in vocab else "<UNKNOWN>")
                supervised_data = np.append(supervised_data, "<end>")
                supervised_data = np.append(["<start>"], supervised_data)
                tags = np.append(tags, "<END>")
                tags = np.append(["<START>"], tags)
                supervised_data = np.array(supervised_data)
                accuracy, precision, recall, homo_score, comp_score, v_score = self.evaluate(supervised_data,tags,states)
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                homo_scores.append(homo_score)
                comp_scores.append(comp_score)
                v_scores.append(v_score)
                if current_sentence/sentence_count > 0.1:
                    break
                else:
                    print("Evaluation Progress: " + str((current_sentence / sentence_count) * 100) + "%")
            outputString = str(np.average(accuracies)) + "," + str(np.average(precisions)) + "," + str(np.average(recalls)) + "," + str(np.average(homo_scores)) + "," + str(np.average(comp_scores)) + "," + str(np.average(v_scores)) + "\n"
            eval_file.write(outputString)
            print(outputString)
            # Break if converged
            #if (utils.kl_divergence(np.array(previous_emmission_matrix), np.array(self.emmission_probs)) < convergence_threshold and utils.kl_divergence(np.array(previous_transition_matrix), np.array(self.transition_matrix)) < convergence_threshold):
             #   break
            #if (np.sum([abs(previous_emmission_matrix[i][v] - self.emmission_probs[i][v]) for i in range(len(states)) for v in vocab]) < convergence_threshold) and (np.sum([abs(previous_transition_matrix[i][j] - self.transition_matrix[i][j]) for i in range(len(states)) for j in range(len(states))]) < convergence_threshold):
             #   break
            if curr_iter >= max_iter:
                break
        eval_file.close()

    def get_states_and_vocab(self, path, label, unknown_threshold):
        dataFile = open(path, "r")
        sentences = conllu.parse_incr(dataFile)
        sentence_count = 0
        data = []
        for sentence in sentences:
            sentence_count += 1
            for token in sentence:
                data.append((token["form"], token[label])) 
        #data = [(token["form"], token[label]) for sentence in sentences for token in sentence]
        data = np.array(data, dtype=[("form", "U20"), ("POS","U8")])
        counter = collections.Counter(data["form"])
        frequencies = counter.items()
        states = list(set(data["POS"]))
        frequency_sum = counter.total()
        vocab = set([])
        for (key, value) in frequencies:
            if float(value) / float(frequency_sum) < unknown_threshold:
                vocab.add("<unknown>")
            else:
                vocab.add(key)
        vocab = list(vocab)
        return (states,vocab, sentence_count)

    def log_sum_exp(self, sequence):
        if len(sequence) == 0:
            return 0
        sequence = np.array(sequence)
        c = sequence.max()
        return c + np.log(np.sum(np.exp(sequence - c)))

    def forward_backward(self, untagged_data, states, vocab):
        # Build transition and observation probabilities using forward-backward(Baumm-Welch)
        # To understand the variable names, please consult Jurafsky and Martin pg. 
        # Build vocab and states
        T = untagged_data.size
        N = states.size
        V = vocab.size
        # E step
        alpha = np.transpose(self.forward_probs(untagged_data, states))
        beta = self.backwards_probs(untagged_data,states)
        gamma = self.generate_gamma(untagged_data, alpha, beta, N, T)
        xi = self.generate_xi(untagged_data, alpha, beta, N, T)
        # M step
        for i in range(N):
            den_sum_list = xi[:-1,i,:].flatten()
            den = self.log_sum_exp(den_sum_list)
            for j in range(N):
                num_sum_list = xi[:,i,j]
                self.transition_matrix[i][j] = self.log_sum_exp(num_sum_list) - den

        for j in range(N):
            den_sum_list = gamma[:,j]
            den = self.log_sum_exp(den_sum_list)
            for k in set(untagged_data):
                num_sum_list = [gamma[t,j] if k == untagged_data[t] else 0 for t in range(T)]
                self.emmission_probs[j][k] = self.log_sum_exp(num_sum_list) - den
    
    def generate_xi(self, data, alpha, beta, N, T):
        xi = np.zeros((T,N,N))
        for t in range(T-1):
            sum_list = []
            for i in range(N):
                for j in range(N):
                    num = alpha[t,i] + self.transition_matrix[i][j] + self.emmission_probs[j][data[t+1]] + beta[t+1, j]
                    sum_list.append(num)
                    xi[t][i][j] = num
            den = self.log_sum_exp(sum_list)

            for i in range(N):
                for j in range(N):
                    xi[t][i][j] -= den
        return xi

    def generate_gamma(self, data, alpha, beta, N, T):
        gamma = np.zeros((T,N))
        for t in range(T-1):
            sum_list = []
            for j in range(N):
                num = alpha[t,j] + beta[t,j]
                sum_list.append(num)
                gamma[t,j] = num
            den = self.log_sum_exp(sum_list)

            for j in range(N):
                gamma[t,j] -= den
        return gamma


    def forward_probs(self, observations, state_sequence):
        N = state_sequence.size
        T = observations.size
        #NTS probably an off by one error to so with the added start and finish states and tokens
        forward = np.zeros((N,T))
        # initialisation step
        for s in range(1, N):
            forward[s, 1] = self.transition_matrix[0][s] + self.emmission_probs[s][observations[1]]
        # recursion step
        for t in range(1, T):
            for s in range(1,N):
                sum_list = []
                for s2 in range(1,N):
                    sum_list.append(forward[s2,t-1] + self.transition_matrix[s2][s])
                forward[s,t] = self.log_sum_exp(sum_list) + self.emmission_probs[s][observations[t]]
        return forward

    def backwards_probs(self, observations, state_sequence):
        #NTS again there may be some off by one error in here
        N = state_sequence.size
        T = observations.size
        backwards = np.zeros((T,N))
        #Initialisation step
        for i in range(1, N):
            backwards[T-1, i] = self.transition_matrix[i][N-1]
        #Recursion step
        for t in reversed(range(1, T-1)):
            for i in range(1, N):
                sum_list = []
                for j in range(1, N):
                    sum_list.append(self.transition_matrix[i][j] * self.emmission_probs[j][observations[t+1]] * backwards[t+1 ,j])
                backwards[t,i] = self.log_sum_exp(sum_list)
        return backwards
    
    def evaluate(self, emission_sequence, tag_sequence, states):
        true_transition_matrix = [[np.exp(self.transition_matrix[i][j]) for i in range(len(states))] for j in range(len(states))]
        true_emission_probs = [{v: np.exp(self.emmission_probs[i][v]) for v in self.emmission_probs[0].keys()} for i in range(len(states))]
        predicted_tags = utils.viterbi(np.array(emission_sequence), len(states), np.array(true_transition_matrix), true_emission_probs)
        predicted_tags = [states[i] for i in predicted_tags]
        if len(tag_sequence) == 0:
            return 0,0,0
        # Calcuate accuracy, precision, recall and F-score and report
        try:
            accuracy = sum([1 if tag_sequence[i] == predicted_tags[i] else 0 for i in range(len(tag_sequence))]) / len(predicted_tags)
        except:
            print(tag_sequence)
            print(predicted_tags)
        precisions = []
        recalls = []
        for state in states:
            TP = 0.0
            FP = 0.0
            FN = 0.0
            for i in range(len(predicted_tags)):
                if predicted_tags[i] == state and tag_sequence[i] == state:
                    TP += 1.0
                elif predicted_tags[i] == state and tag_sequence[i] != state:
                    FP += 1.0
                elif predicted_tags[i] != state and tag_sequence[i] == state:
                    FN += 1.0
            if TP+FP == 0.0:
                precisions.append(0.0)
            else:
                precisions.append(TP/(TP+FP))
            if TP+FN == 0.0:
                recalls.append(0.0)
            else:
                recalls.append(TP/(TP+FN))

        
        homo_score, comp_score, v_score = utils.calculate_v_measure(tag_sequence, [list(states).index(str(predicted_tags[i])) for i in range(len(predicted_tags))])
        return accuracy, np.average(precisions), np.average(recalls), homo_score, comp_score, v_score

###
### Subtask 2: K-means clustering 
###

def kMeansClustering(path, k, test, postags):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataFile = open(path, "r")
    sentences = conllu.parse_incr(dataFile)
    datapoints = []
    curr_sentence = 0
    true_clusters_unadj = []
    for sentence in sentences:
        curr_sentence += 1
        if curr_sentence >= 100 and test:
            break
        # Read in sentence and add special BERT tokens [CLS] and [SEP]
        data = "[CLS] "
        true_clusters_unadj = ["[CLS]"]
        for token in sentence:
            data = data + token["form"] + " "
            true_clusters_unadj.append(token[postags])
        print(sentence)
        data = data + " [SEP]"
        true_clusters_unadj.append("[SEP]")
        # Tokenize read data
        true_clusters = []
        last_tag = ""
        last_tag_index = 0
        tokenized_data = tokenizer.tokenize(data)
        for i in range(len(tokenized_data)):
            if tokenized_data[i][0:2]== "##":
                true_clusters.append(last_tag)
            #elif  tokenized_data[i][0] == "." or tokenized_data[i][0] == "'" or tokenized_data[i][0] == "," or tokenized_data[i][0] == '-':
            #    true_clusters.append("PUNCT")
            elif tokenized_data[i][0].isalpha():
                last_tag = true_clusters_unadj[last_tag_index]
                last_tag_index += 1
                true_clusters.append(last_tag)
            else:
                true_clusters.append("PUNCT")
            print(str(tokenized_data[i]) + " = " + str(last_tag))
        # Map token strings to their BERT vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_data)
        # Generate the ids for the segment, indicating all tokens belong to the same sentence
        segments_ids = [1] * len(tokenized_data)
        # Convert token indices and segments to tensors for BERT's use
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        # Load pre-trained model and place in evaluation mode
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        model.eval()
        with torch.no_grad():
            # Evaluate the sentence and extract the hidden states of the model
            outputs = model(tokens_tensor, segments_tensor)
            hidden_states = outputs[2]
            # Manipulate the hidden states so we have the values grouped by token
            token_embeddings = torch.stack(hidden_states, dim = 0)
            token_embeddings = torch.squeeze(token_embeddings, dim = 1)
            token_embeddings = token_embeddings.permute(1,0,2)
            # We use these hidden states to generate the embeddings - here I choose to use the last four layers and combine by summing
            token_vecs = []
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim = 0)
                token_vecs.append(sum_vec)
            # Feed the sentence back to the overall dataset
            datapoints = datapoints + token_vecs
    #
    #   Perform the K-means clustering
    #
    # Initialise k centroids at random points
    centroids = [[random.random() for j in range(int(len(datapoints[0])))] for i in range(k)]
    final_assignments = []
    # For 30 iterations
    for z in range(30):
        assignments = []
        for datapoint in datapoints:
            # Calculate the distances from each centroid
            distances = []
            for centroid in centroids:
                distances.append(utils.euclidean_distance(np.array(datapoint), np.array(centroid)))
            # Assign each datapoint to it's closest node
            # assignment[i] will give the centroid index of the closest centroid to datapoint i
            assignment_dist = distances[0]
            assignment_index = 0
            for j in range(len(distances)):
                if(distances[j] < assignment_dist):
                    assignment_dist = distances[j]
                    assignment_index = j
            assignments.append(assignment_index)
        # Calculate the centre of each cluster
        sums = [0] * k
        counts = [0] * k
        for i in range(len(assignments)):
            counts[assignments[i]] = counts[assignments[i]] + 1
            sums[assignments[i]] = sums[assignments[i]] + datapoints[i]
        # if there are no datapoints assigned to a centroid, re initialise in a fresh random place
        centroids = [sums[i]/counts[i] if counts[i] > 0 else [random.random() for j in range(int(len(datapoints[0])))]  for i in range(k)]
        final_assignments = assignments
    # Evaluate the model and report the values
    # Sort predicted clusters into lists for evaluation
    # Use scipy to generate "true" clusters
    # Each cluster represents a POS so to obtain true cluster, cluster all words in the datapoints of the same POS
    # Generate scores and write to file
    eval_file = open("KMCeval.txt", "w")
    homo_score, comp_score, v_score = utils.calculate_v_measure(true_clusters, final_assignments[:len(true_clusters)])
    eval_file.write(str(homo_score) + " , " + str(comp_score) + " , " + str(v_score))
    eval_file.close()

MyTagger = HMMPOSTagger("./ptb-train.conllu", "upos", 0.00005, 5, True, 5)

#kMeansClustering("./ptb-train.conllu", 45, True, "upos")