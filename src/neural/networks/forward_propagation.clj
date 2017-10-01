(ns neural.networks.forward-propagation
  (:require [clojure.core.matrix :as matrix])
  (:use [logistic-regression :only (sigmoid)]))

(defn- calc-theta-vec [structure]
  (reduce
    #(conj %1 (matrix/new-matrix (second %2) (inc (first %2))))
    []
    (partition 2 1 structure)))

;a must be a n*1 activation matrix
(defn- add-bias-to-activation [a]
  (matrix/join-along 0 [[1]] a))

;Calculate next layer's activation then attach bias unit.
;Given activation must include the bias unit and is a (n+1)*1 matrix.
(defn- calc-next-activation [activation theta]
  (let [z (matrix/mmul theta activation)
        a (matrix/emap sigmoid z)]
    (add-bias-to-activation a)))

;x must be a vector
(defn forward-propagation [x structure]
  (reduce
    calc-next-activation
    (add-bias-to-activation (matrix/transpose [x]))
    (calc-theta-vec structure)))