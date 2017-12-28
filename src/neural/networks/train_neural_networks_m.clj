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

(defn- create-theta-matrix [structure]
  (map #(matrix/new-matrix (second %) (inc (first %)))
       (partition 2 1 structure)))

(defn- gen-random-epsilon []
  (- (* 2 (rand) epsilon) epsilon))

(defn- init-theta [structure]
  (map #(matrix/emap gen-random-epsilon %) (create-theta-matrix structure)))

(defn- regularize-big-delta [m, lambda]
  (fn [big-delta theta]
    (let [processed-big-delta (matrix/mul (/ 1 m) big-delta)
          last-column-index (dec (matrix/column-count processed-big-delta))
          big-delta-first-column (matrix/submatrix processed-big-delta 1 [0 1])
          big-delta-rest-columns (matrix/submatrix processed-big-delta 1 [1 last-column-index])
          theta-rest-columns (matrix/submatrix theta 1 [1 last-column-index])
          processed-theta (matrix/mul (/ lambda m) theta-rest-columns)]
      (matrix/join-along 1 big-delta-first-column (matrix/add big-delta-rest-columns processed-theta)))))

(defn- calc-big-deltas [X Y thetas]
  (let [activations (fp/calc-activation-seq X thetas)]
    (bp/calc-big-deltas thetas activations Y)))

;TODO: add validation
;X matrix of training set
;Y matrix of result set
;theta-seq can be initial theta sequence
;return '(D1 D2 ... D(L-1))
(defn- calc-theta-directive [X Y thetas lambda]
  (let [big-deltas (calc-big-deltas X Y thetas)
        calc-theta-descent (regularize-big-delta (matrix/row-count X) lambda)]
    (map calc-theta-descent big-deltas thetas)))

(defn- perform-one-step-theta-directive [alpha theta one-step-directive]
  (let [temp (map #(matrix/mul alpha %) one-step-directive)]
    (map matrix/sub theta temp)))

;TODO: this way may lead to local optimized point. should find out alternative of fmincg instead.
(defn perform-batch-gradient-decent [X Y alpha lambda iter]
  (loop [i 0
         Theta (init-theta neural-networks-structure)]
    (if (= i iter)
      Theta
      (recur
        (inc i)
        (perform-one-step-theta-directive alpha Theta (calc-theta-directive X Y Theta lambda))))))