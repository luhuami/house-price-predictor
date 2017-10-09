(ns neural.networks.backward-propagation-m
  (:require [clojure.core.matrix :as matrix]))

(defn- create-ones [dimension-vec]
  (matrix/fill (matrix/new-matrix (first dimension-vec) (last dimension-vec)) 1))

(defn- calc-delta-for-one-layer [delta-list theta-activation-pair]
  (let [next-layer-delta (first delta-list)
        theta (first theta-activation-pair)
        activation (second theta-activation-pair)
        temp (matrix/mmul next-layer-delta theta)
        ones (create-ones (matrix/shape activation))]
    (conj delta-list (matrix/emul temp activation (matrix/sub ones activation)))))

;drop the theta1 as don't need to calc delta-1.
;drop aL as it's already used to calculate delta-L.
(defn- generate-theta-activation-pairs [theta-seq activation-seq]
  (partition 2 (interleave (drop 1 theta-seq) (drop-last activation-seq))))

;Length of both theta-seq and activation-seq should be L-1
(defn- calc-deltas [theta-seq activation-seq Y]
  (reduce
    calc-delta-for-one-layer
    (list (matrix/sub (last activation-seq) Y))
    (reverse (generate-theta-activation-pairs theta-seq activation-seq))))

(defn- remove-bias [delta-matrix]
  (matrix/submatrix delta-matrix 1 [1 (dec (matrix/row-count delta-matrix))]))

(defn- remove-bias-for-deltas [delta-list]
  (map-indexed
    #((if (< %1 (dec (count delta-list))) (remove-bias %2)))
    delta-list))

(defn- calc-big-delta [next-layer-delta activation]
  (matrix/mmul (matrix/transpose next-layer-delta) activation))

;L is the total number of layers including input layer and output layer
;activation-seq '(a1 a2 a3 ... aL)
;theta-seq '(theta1 theta2 ... theta(L-1))
;return '(big-delta1 big-delta2 ... big-delta(L-1))
(defn calc-deltas-for-all-training-data [theta-seq activation-seq Y]
  (let [delta-list (remove-bias-for-deltas (calc-deltas theta-seq (rest activation-seq) Y))
        a (drop-last activation-seq)]
    (map calc-big-delta delta-list a)))