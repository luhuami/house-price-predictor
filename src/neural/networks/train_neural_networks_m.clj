(ns neural.networks.train-neural-networks-m
  (:require [clojure.core.matrix :as matrix]
            [neural.networks.forward-propagation-m :as fp]
            [neural.networks.backward-propagation-m :as bp]))

;number of neural in each layer excluding bias node. layer num starts from 1.
;[400 25 10] represents a 3 layers neural networks.
;input layer (1st layer) has 400 nodes (features). hidden layer (2nd layer) has 25 nodes.
;output layer has 10 nodes (output).
(def neural-networks-structure [400 25 10])

;used in standford machine learning course
(def epsilon 0.12)

(defn- init-based-on-structure [structure]
  (reverse (reduce
             #(conj %1 (matrix/new-matrix (second %2) (inc (first %2))))
             (list)
             (partition 2 1 structure))))

(defn- gen-random-epsilon []
  (- (* 2 (rand) epsilon) epsilon))

(defn- init-theta [structure]
  (map #(matrix/emap gen-random-epsilon %) (init-based-on-structure structure)))

(defn- process-big-delta [m, lambda]
  (fn process-delta [big-delta theta-matrix]
    (let [processed-big-delta (matrix/mul (/ 1 m) big-delta)
          last-column-index (dec (matrix/column-count processed-big-delta))
          big-delta-first-column (matrix/submatrix processed-big-delta 1 [0 1])
          big-delta-rest-columns (matrix/submatrix processed-big-delta 1 [1 last-column-index])
          theta-rest-columns (matrix/submatrix theta-matrix 1 [1 last-column-index])
          processed-theta (matrix/mul (/ lambda m) theta-rest-columns)]
      (matrix/join-along 1 big-delta-first-column (matrix/add big-delta-rest-columns processed-theta)))))

(defn- calc-big-deltas [structure X Y thetas]
  (let [initial-deltas (init-based-on-structure structure)
        activations (fp/calc-activation-seq X thetas)]
    (bp/calc-deltas-for-all-training-data thetas activations Y)))

;TODO: add validation
;X matrix of training set
;Y matrix of result set
;theta-seq can be initial theta sequence
;return '(D1 D2 ... D(L-1))
(defn calc-one-step-theta-directive [X Y thetas lambda]
  (let [big-deltas (calc-big-deltas neural-networks-structure X Y thetas)
        calc-theta-derivative (process-big-delta (matrix/row-count X) lambda)]
    (map calc-theta-derivative big-deltas thetas)))