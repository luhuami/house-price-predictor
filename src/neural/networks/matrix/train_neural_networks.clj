(ns neural.networks.matrix.train-neural-networks
  (:require [clojure.core.matrix :as matrix]
            [neural.networks.matrix.forward-propagation :as fp]
            [neural.networks.matrix.backward-propagation :as bp]))

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

(defn- gen-random-epsilon [_]
  (- (* 2 (rand) epsilon) epsilon))

(defn- init-theta [structure]
  (map #(matrix/emap gen-random-epsilon %) (create-theta-matrix structure)))

(defn- apply-theta-directive [alpha theta one-step-directive]
  (let [temp (map #(matrix/mul alpha %) one-step-directive)]
    (map matrix/sub theta temp)))

;TODO: using gradient decent may lead to local optimized point. should find out an alternative of fmincg.
;TODO: add validation
;X matrix of training set
;Y matrix of result set
;iter number of iteration
(defn train-theta [X Y alpha lambda iter]
  (loop [i 0
         theta-seq (init-theta neural-networks-structure)]
    (if (= i iter)
      theta-seq
      (recur
        (inc i)
        (apply-theta-directive alpha theta-seq (bp/calc-theta-directives X Y theta-seq lambda))))))