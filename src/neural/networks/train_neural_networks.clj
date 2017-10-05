(ns neural.networks.train-neural-networks
  (:require [clojure.core.matrix :as matrix]
            [neural.networks.forward-propagation :as fp]
            [neural.networks.backward-propagation :as bp]))

;number of neural in each layer excluding bias node. layer num starts from 1.
;[400 25 10] represents a 3 layers neural networks.
;input layer (1st layer) has 400 nodes (features). hidden layer (2nd layer) has 25 nodes.
;output layer has 10 nodes (output).
(def neural-networks-structure [400 25 10])

(defn- calc-theta-seq [structure]
  (reverse (reduce
             #(conj %1 (matrix/new-matrix (second %2) (inc (first %2))))
             (list)
             (partition 2 1 structure))))

(defn- calc-delta [theta-seq]
  (fn [delta-seq train-pair]
    (let [activation-seq (fp/calc-activation-seq (first train-pair) theta-seq)
          new-delta-seq (bp/calc-one-delta-for-all-layers theta-seq activation-seq (last train-pair))]
      (map (matrix/add %1 %2) delta-seq new-delta-seq))))

(defn- process-accumulated-delta [m, lambda]
  (fn process-delta [delta-matrix theta-matrix]
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
(defn- calc-deltas [structure X Y theta-seq]
  (let [initial-delta-seq (calc-theta-seq structure)
        train-pairs (partition 2 (interleave (matrix/rows X) (matrix/rows Y)))]
    (reduce
      (calc-delta theta-seq)
      initial-delta-seq
      train-pairs)))

;TODO: add validation
(defn calc-one-step-theta-directive [X Y theta-seq lambda]
  (let [accumulated-delta-seq (calc-deltas neural-networks-structure X Y theta-seq)
        combine-fun (process-accumulated-delta (matrix/row-count X) lambda)]
    (map combine-fun accumulated-delta-seq theta-seq)))