(ns utils.unroll
  (:require [clojure.core.matrix :as matrix]))

(defn- calc-theta-num-for-layer [this-layer-theta-num next-layer-theta-num]
  (* (inc this-layer-theta-num) next-layer-theta-num))

(defn- calc-total-theta-num [structure]
  (reduce #(+ %1 (calc-theta-num-for-layer (first %2) (second %2)))
          0
          (partition 2 1 structure)))

(defn get-theta-matrix [layer-num theta-vec structure]
  (if (and (< layer-num (count structure)) (< 0 layer-num))
    (let [previous-layers-theta-num (calc-total-theta-num (subvec structure 0 layer-num))
          theta-num-this-layer (get structure (dec layer-num))
          theta-num-next-layer (get structure layer-num)
          theta-num (calc-theta-num-for-layer theta-num-this-layer theta-num-next-layer)
          theta-vec (subvec theta-vec previous-layers-theta-num (+ theta-num previous-layers-theta-num))]
      (matrix/reshape theta-vec [theta-num-next-layer (inc theta-num-this-layer)]))))