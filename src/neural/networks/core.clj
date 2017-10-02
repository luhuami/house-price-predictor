(ns neural.networks.core
  (:require [clojure.core.matrix :as matrix]
            [neural.networks.forward-propagation :as fp]))

;number of neural in each layer excluding bias node. layer num starts from 1.
;[400 25 10] represents a 3 layers neural networks.
;input layer (1st layer) has 400 nodes (features). hidden layer (2nd layer) has 25 nodes.
;output layer has 10 nodes (output).
(def neural-networks-structure [400 25 10])

(def theta-vec [])

(defn- calc-theta-vec [structure]
  (reduce
    #(conj %1 (matrix/new-matrix (second %2) (inc (first %2))))
    []
    (partition 2 1 structure)))

(defn calc [x structure]
  (let[theta-vec (calc-theta-vec structure)
       activation-vec (fp/calc-activation-vec x theta-vec)]
    ))