(ns neural.networks.backward-propagation
  (:require [clojure.core.matrix :as matrix]))

;TODO: generalize this function
(defn- create-ones [dimension-vec]
  (let [m (matrix/new-matrix (first dimension-vec) (last dimension-vec))]
    (matrix/fill m 1)))

(defn- calc-layer-delta [delta-list theta-activation-pair]
  (let [next-layer-delta (first delta-list)
        theta (first theta-activation-pair)
        activation (second theta-activation-pair)
        temp (matrix/mmul (matrix/transpose theta) next-layer-delta)
        ones (create-ones (matrix/shape activation))]
    (conj delta-list (matrix/emul temp activation (matrix/sub ones activation)))))

;drop the theta1 as don't need to calc delta1
;drop al as it's not needed to calc a(l-1)
(defn- generate-theta-activation-pairs [theta-seq activation-seq]
  (partition 2 (interleave (drop 1 theta-seq) (drop-last activation-seq))))

;Length of both theta-seq and activation-seq should be l-1
(defn- calc-delta-list [theta-seq activation-seq y]
  (reduce
    calc-layer-delta
    (list (matrix/sub (last activation-seq) y))
    (reverse (generate-theta-activation-pairs theta-seq activation-seq))))

(defn- remove-bias-delta [delta-matrix]
  (matrix/submatrix delta-matrix 0 [1 (dec (matrix/row-count delta-matrix))]))

(defn- remove-bias-unit-delta [delta-list]
  (map-indexed
    #((if (< %1 (dec (count delta-list))) (remove-bias-delta %2)))
    delta-list))

(defn- calc-one-step-gradient-decent [next-layer-delta activation]
  (matrix/mmul next-layer-delta (matrix/transpose activation)))

;activation-seq doesn't have a1 since it's x itself '(a2 a3 ... al)
;theta-seq is '(theta1 theta2 ... theta(l-1))
(defn calc-one-step-gradient-decent-for-all-layers [theta-seq activation-seq y]
  (let [raw-delta-list (calc-delta-list theta-seq activation-seq y)
        delta-list (remove-bias-unit-delta raw-delta-list)
        a (drop-last activation-seq)]
    (map calc-one-step-gradient-decent delta-list a)))