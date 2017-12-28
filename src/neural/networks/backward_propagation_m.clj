(ns neural.networks.backward-propagation-m
  (:require [clojure.core.matrix :as matrix]
            [utils.matrix :as utils]))

(defn- calc-delta [next-layer-delta theta-activation-pair]
  (let [theta (first theta-activation-pair)
        activation (second theta-activation-pair)
        temp (matrix/mmul next-layer-delta theta)
        ones (utils/create-matrix-with-value (matrix/shape activation) 1)]
    (matrix/emul temp activation (matrix/sub ones activation))))

;drop theta1 and a1 as don't need to calc delta1
;drop aL as it's already used to calculate deltaL
(defn- generate-theta-activation-pairs [theta-seq activations]
  (partition 2 (interleave (rest theta-seq) (drop-last (rest activations)))))

;Length of theta-seq should be L-1
;Length of activation-seq should be L
;returns deltaL, delta(L-1) ... delta2
(defn- calc-deltas [theta-seq activation-seq Y]
  (reductions
    calc-delta
    (matrix/sub (last activation-seq) Y)
    (reverse (generate-theta-activation-pairs theta-seq activation-seq))))

(defn- remove-bias-for-deltas [delta-list]
  (conj (map utils/remove-first-column (rest delta-list)) (first delta-list)))

(defn- calc-big-delta [next-layer-delta activation]
  (matrix/mmul (matrix/transpose next-layer-delta) activation))

;L is the total number of layers including input layer and output layer
;activation-seq '(a1 a2 a3 ... aL)
;theta-seq '(theta1 theta2 ... theta(L-1))
;return '(big-delta1 big-delta2 ... big-delta(L-1))
(defn calc-big-deltas [theta-seq activation-seq Y]
  (let [delta-list (remove-bias-for-deltas (calc-deltas theta-seq activation-seq Y))]
    (map calc-big-delta (reverse delta-list) (drop-last activation-seq))))