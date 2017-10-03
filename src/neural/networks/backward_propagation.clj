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

(defn- generate-theta-activation-pair [theta-vec activation-vec]
  (let [t-a-pair (partition 2 (interleave theta-vec (drop-last activation-vec)))]
    (drop-last (reverse t-a-pair))))

;Length of theta-vec should be l-1.Length of activation-vec should be l,
(defn- calc-delta-list [theta-vec activation-vec y]
  (reduce
    calc-layer-delta
    (list (matrix/sub (last activation-vec) y))
    (generate-theta-activation-pair theta-vec activation-vec)))

;TODO: delta-matrix a matrix or vector?
(defn- remove-bias-delta [delta-matrix]
  (matrix/submatrix delta-matrix 0 [1 (dec (matrix/row-count delta-matrix))]))

(defn- remove-bias-unit-delta [delta-list]
  (map-indexed
    #((if (< %1 (dec (count delta-list))) (remove-bias-delta %2)))
    delta-list))

(defn- calc-one-step-gradient-decent [next-layer-delta activation]
  (matrix/mmul next-layer-delta (matrix/transpose activation)))

(defn calc-one-step-gradient-decent-for-all-layers [theta-vec activation-vec y]
  (let [raw-delta-list (calc-delta-list theta-vec activation-vec y)
        delta-list (remove-bias-unit-delta raw-delta-list)
        a (drop-last activation-vec)]
    (map calc-one-step-gradient-decent delta-list a)))