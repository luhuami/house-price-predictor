(ns neural.networks.backward-propagation
  (:require [clojure.core.matrix :as matrix]))

;TODO: generalize this function
(defn- create-ones [dimension-vec]
  (let [m (matrix/new-matrix (first dimension-vec) (last dimension-vec))]
    (matrix/fill m 1)))

(defn- calc-delta1 [next-layer-delta theta activation]
  (let [temp (matrix/mmul (matrix/transpose theta) next-layer-delta)
        ones (create-ones (matrix/shape activation))]
    (matrix/emul temp activation (matrix/sub ones activation))))

;TODO:merge with above function?
(defn- calc-delta [delta-vec theta-activation-pair]
  (let [next-layer-delta (last delta-vec)
        this-layer-delta (calc-delta1 next-layer-delta (first theta-activation-pair) (second theta-activation-pair))]
    (conj delta-vec this-layer-delta)))

(defn backward-propagation [activation-vec theta-vec, y]
  (reverse (reduce
             calc-delta
             [(matrix/sub (last activation-vec) y)]
             (partition 2 (interleave theta-vec (drop-last activation-vec))))))