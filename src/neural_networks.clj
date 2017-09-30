(ns neural-networks
  (:require [clojure.core.matrix :as matrix]
            [clojure.math.numeric-tower :as math]))

;number of neural in each layer excluding bias node. layer num starts from 1.
;[400 25 10] represents a 3 layers neural networks.
;input layer (1st layer) has 400 nodes (features). hidden layer (2nd layer) has 25 nodes.
;output layer has 10 nodes (output).
(def neural-networks-structure [400 25 10])

(def theta [])

(defn- calc-theta-num-for-layer [this-layer-theta-num next-layer-theta-num]
  (* (inc this-layer-theta-num) next-layer-theta-num))

(defn- calc-total-theta-num [structure]
  (reduce #(+ %1 (calc-theta-num-for-layer (first %2) (second %2)))
          0
          (partition 2 1 structure)))

(defn- init-theta [structure]
  (vec (repeat (calc-total-theta-num structure) 0)))

(defn- get-theta-matrix [layer-num theta-vec structure]
  (if (and (< layer-num (count structure)) (< 0 layer-num))
    (let [previous-layers-theta-num (calc-total-theta-num (subvec structure 0 layer-num))
          theta-num-this-layer (get structure (dec layer-num))
          theta-num-next-layer (get structure layer-num)
          theta-num (calc-theta-num-for-layer theta-num-this-layer theta-num-next-layer)
          theta-vec (subvec theta-vec previous-layers-theta-num (+ theta-num previous-layers-theta-num))]
      (matrix/reshape theta-vec [theta-num-next-layer (inc theta-num-this-layer)]))))

(defn- sigmoid [z]
  (->> z
       (math/expt Math/E)
       (/ 1)
       (+ 1)
       (/ 1)))

;calculate next layer's activation with bias unit.
; Given activation should include bias unit so it should be (n+1)*1.
(defn- calc-next-activation [activation theta]
  (let [z (matrix/mmul theta activation)
        a (matrix/emap sigmoid z)]
    (matrix/join-along 1 [1] a)))

(defn- convert-x [x-vec]
  (matrix/transpose [x-vec]))

(defn- forward-propagation [x theta]
  )