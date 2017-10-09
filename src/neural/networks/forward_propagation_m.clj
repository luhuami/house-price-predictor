(ns neural.networks.forward-propagation-m
  (:require [clojure.core.matrix :as matrix])
  (:use [logistic-regression :only (sigmoid)]))

;TODO: duplicated
(defn- create-ones [dimension-vec]
  (matrix/fill (matrix/new-matrix (first dimension-vec) (last dimension-vec)) 1))

(defn- add-bias-to-activation [a]
  (matrix/join-along 1 (create-ones [(matrix/row-count a) 1]) a))

;Calculate next layer's activation, attach bias unit and save it to a vector.
;Given activations must include the bias unit and is a m*(n+1) matrix.
(defn- calc-next-activation [activation-vec theta]
  (let [z (matrix/mmul (last activation-vec) (matrix/transpose theta))
        a (matrix/emap sigmoid z)]
    (conj activation-vec (add-bias-to-activation a))))

;X is a m*n matrix
;Return [a1 a2 a3 ... al]
(defn calc-activation-seq [X theta-seq]
  (reduce
    calc-next-activation
    [(add-bias-to-activation X)]
    theta-seq))

;TODO: no need to store all activations in this case
(defn calc [x theta-seq]
  (last (calc-activation-seq x theta-seq)))