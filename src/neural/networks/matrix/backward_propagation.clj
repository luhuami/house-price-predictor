(ns neural.networks.matrix.backward-propagation
  (:require [clojure.core.matrix :as matrix]
            [utils.matrix :as utils]))

(defn- calc-small-delta [next-layer-delta theta-activation-pair]
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
(defn- calc-small-deltas [theta-seq activation-seq Y]
  (reductions
    calc-small-delta
    (matrix/sub (last activation-seq) Y)
    (reverse (generate-theta-activation-pairs theta-seq activation-seq))))

(defn- remove-bias-for-deltas [delta-list]
  (conj (map utils/remove-first-column (rest delta-list)) (first delta-list)))

(defn- calc-big-delta [next-layer-delta activation]
  (matrix/mmul (matrix/transpose next-layer-delta) activation))

(defn- calc-big-deltas [theta-seq activation-seq Y]
  (let [small-delta-seq (remove-bias-for-deltas (calc-small-deltas theta-seq activation-seq Y))]
    (map calc-big-delta (reverse small-delta-seq) (drop-last activation-seq))))

(defn- regularize-element [m lambda]
  (fn [index big-delta theta]
    (let [d (/ big-delta m)]
      (if (not= 0 (last index))
        (+ d (* lambda (/ theta m)))
        d))))

;Given L is the total number of layers including input layer and output layer
;activation-seq '(a1 a2 a3 ... aL)
;theta-seq '(theta1 theta2 ... theta(L-1))
;return '(big-delta1 big-delta2 ... big-delta(L-1))
(defn calc-theta-directives [theta-seq activation-seq Y lambda]
  (let [big-delta-seq (calc-big-deltas theta-seq activation-seq Y)
        regularize (regularize-element (matrix/row-count Y) lambda)]
    (map #(matrix/emap-indexed regularize %1 %2) big-delta-seq theta-seq)))