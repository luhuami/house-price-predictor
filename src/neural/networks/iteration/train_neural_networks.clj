(ns neural.networks.iteration.train-neural-networks
  (:require [clojure.core.matrix :as matrix]
            [neural.networks.iteration.forward-propagation :as fp]
            [neural.networks.iteration.backward-propagation :as bp]))

;number of neural in each layer excluding bias node. layer num starts from 1.
;[400 25 10] represents a 3 layers neural networks.
;input layer (1st layer) has 400 nodes (features). hidden layer (2nd layer) has 25 nodes.
;output layer has 10 nodes (output).
(def neural-networks-structure [400 25 10])

;used in standford machine learning course
(def epsilon 0.12)

(defn- create-theta-matrix [structure]
  (map
    #(matrix/new-matrix (second %) (inc (first %)))
    (partition 2 1 structure)))

(defn- gen-random-theta []
  (- (* 2 (rand) epsilon) epsilon))

(defn- init-theta [structure]
  (map #(matrix/emap gen-random-theta %) (create-theta-matrix structure)))

(defn- calc-delta [theta-seq]
  (fn [accumulated-deltas train-data]
    (let [activations (fp/calc-activation-seq (first train-data) theta-seq)
          new-deltas (bp/calc-deltas-for-one-training-data theta-seq activations (last train-data))]
      (map (matrix/add %1 %2) accumulated-deltas new-deltas))))

(defn- regularize-accumulated-delta [m, lambda]
  (fn [delta-matrix theta-matrix]
    (let [processed-delta-matrix (matrix/mul (/ 1 m) delta-matrix)
          last-column-index (dec (matrix/column-count processed-delta-matrix))
          first-column-delta (matrix/submatrix processed-delta-matrix 1 [0 1])
          rest-columns-delta (matrix/submatrix processed-delta-matrix 1 [1 last-column-index])
          rest-columns-theta (matrix/submatrix theta-matrix 1 [1 last-column-index])
          processed-theta (matrix/mul (/ lambda m) rest-columns-theta)]
      (matrix/join-along 1 first-column-delta (matrix/add rest-columns-delta processed-theta)))))

;X matrix of training set
;Y matrix of result set
;theta-seq can be initial theta sequence
;TODO: X Y can be vector of vectors no need to be a matrix?
(defn- calc-deltas [structure X Y thetas]
  (let [initial-deltas (create-theta-matrix structure)
        train-pairs (partition 2 (interleave (matrix/rows X) (matrix/rows Y)))]
    (reduce
      (calc-delta thetas)
      initial-deltas
      train-pairs)))

;TODO: add validation
(defn calc-one-step-theta-directive [X Y thetas lambda]
  (let [accumulated-deltas (calc-deltas neural-networks-structure X Y thetas)
        calc-theta-descent (regularize-accumulated-delta (matrix/row-count X) lambda)]
    (map calc-theta-descent accumulated-deltas thetas)))