(ns neural.networks.backward-propagation
  (:require [clojure.core.matrix :as matrix]))

(defn- create-ones [dimension-vec]
  (matrix/fill (matrix/new-matrix (first dimension-vec) (last dimension-vec)) 1))

(defn- calc-delta-for-one-layer [delta-list theta-activation-pair]
  (let [next-layer-delta (first delta-list)
        theta (first theta-activation-pair)
        activation (second theta-activation-pair)
        temp (matrix/mmul (matrix/transpose theta) next-layer-delta)
        ones (create-ones (matrix/shape activation))]
    (conj delta-list (matrix/emul temp activation (matrix/sub ones activation)))))

;drop the theta1 as don't need to calc delta1
;drop aL as it's already used to calculate delta-L
(defn- generate-theta-activation-pairs [theta-seq activation-seq]
  (partition 2 (interleave (drop 1 theta-seq) (drop-last activation-seq))))

;Length of both theta-seq and activation-seq should be L-1
(defn- calc-deltas [theta-seq activation-seq y]
  (reduce
    calc-delta-for-one-layer
    (list (matrix/sub (last activation-seq) y))
    (reverse (generate-theta-activation-pairs theta-seq activation-seq))))

(defn- remove-bias [delta-matrix]
  (matrix/submatrix delta-matrix 0 [1 (dec (matrix/row-count delta-matrix))]))

(defn- remove-bias-for-deltas [delta-list]
  (map-indexed
    #((if (< %1 (dec (count delta-list))) (remove-bias %2)))
    delta-list))

(defn- calc-one-step-gradient-decent [next-layer-delta activation]
  (matrix/mmul next-layer-delta (matrix/transpose activation)))

;activation-seq '(a1 a2 a3 ... aL)
;theta-seq is '(theta1 theta2 ... theta(L-1))
(defn calc-deltas-for-one-training-data [theta-seq activation-seq y]
  (let [delta-list (remove-bias-for-deltas (calc-deltas theta-seq (rest activation-seq) y))
        a (drop-last activation-seq)]
    (map calc-one-step-gradient-decent delta-list a)))