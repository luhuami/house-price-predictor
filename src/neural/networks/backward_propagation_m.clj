(ns neural.networks.backward-propagation-m
  (:require [clojure.core.matrix :as matrix]
            [utils.matrix :as utils]))

(defn- calc-delta-for-one-layer [delta-list theta-activation-pair]
  (let [next-layer-delta (first delta-list)
        theta (first theta-activation-pair)
        activation (second theta-activation-pair)
        temp (matrix/mmul next-layer-delta theta)
        ones (utils/create-matrix-with-value (matrix/shape activation) 1)]
    (conj delta-list (matrix/emul temp activation (matrix/sub ones activation)))))

;drop the theta-1 and a1 as don't need to calc delta1
;drop aL as it's already used to calculate delta-L.
(defn- generate-theta-activation-pairs [theta-seq activations]
  (partition 2 (interleave (drop 1 theta-seq) (drop-last (rest activations)))))

;Length of theta-seq should be L-1
;Length of activation-seq should be L
(defn- calc-deltas [theta-seq activation-seq Y]
  (reduce
    calc-delta-for-one-layer
    (list (matrix/sub (last activation-seq) Y))
    (reverse (generate-theta-activation-pairs theta-seq activation-seq))))

(defn- remove-bias-for-deltas [delta-list]
  (map-indexed
    #((if (< %1 (dec (count delta-list))) (utils/remove-first-column %2)))
    delta-list))

(defn- calc-big-delta [next-layer-delta activation]
  (matrix/mmul (matrix/transpose next-layer-delta) activation))

;L is the total number of layers including input layer and output layer
;activation-seq '(a1 a2 a3 ... aL)
;theta-seq '(theta1 theta2 ... theta(L-1))
;return '(big-delta1 big-delta2 ... big-delta(L-1))
(defn calc-deltas-for-all-training-data [theta-seq activation-seq Y]
  (let [delta-list (remove-bias-for-deltas (calc-deltas theta-seq activation-seq Y))]
    (map calc-big-delta delta-list (drop-last activation-seq))))