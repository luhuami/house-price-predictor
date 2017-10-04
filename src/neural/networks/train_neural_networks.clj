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

;X matrix of training set
;Y matrix of result set
;theta-seq can be initial theta sequence
;TODO: X Y can be vector of vecotrs no need to be a matrix?
(defn- calc-deltas [structure X Y theta-seq]
  (let [initial-delta-seq (calc-theta-seq structure)
        train-pairs (partition 2 (interleave (matrix/rows X) (matrix/rows Y)))]
    (reduce
      (calc-delta theta-seq)
      initial-delta-seq
      train-pairs)))

;TODO: divide by m and add lambda theta