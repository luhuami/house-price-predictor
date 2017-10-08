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
;drop al as it's already used to calculate delta-l.
(defn- generate-theta-activation-pairs [theta-seq activation-seq]
  (partition 2 (interleave (drop 1 theta-seq) (drop-last activation-seq))))

;Length of both theta-seq and activation-seq should be l-1
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

;(defn- calc-one-step-gradient-decent [next-layer-delta activation]
;  (matrix/mmul (matrix/transpose next-layer-delta) activation))

;delta and activation are vector.
(defn- create-theta-directive [delta activation]
  (matrix/mmul (matrix/transpose [delta]) [activation]))

(defn- calc-big-delta [next-layer-delta activation]
  (let [matrix-seq (map create-theta-directive (matrix/rows next-layer-delta) (matrix/rows activation))]
    (reduce
      matrix/add
      matrix-seq)))

;activation-seq '(a2 a3 ... al) doesn't have a1 since it's X.
;theta-seq is '(theta1 theta2 ... theta(l-1))
;return '(big-delta1 big-delta2 ... big-delta(l-1))
(defn calc-deltas-for-one-training-data [theta-seq activation-seq Y]
  (let [delta-list (remove-bias-for-deltas (calc-deltas theta-seq activation-seq Y)) ;'(delta2 delta3 ... delta-l)
        ;TODO: should have a1 but not al '(a1 a2 ... a(l-1))
        a (drop-last activation-seq)]
    (map calc-big-delta delta-list a)))