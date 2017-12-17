(ns neural.networks.forward-propagation-m
  (:require [clojure.core.matrix :as matrix]
            [utils.matrix :as utils])
  (:use [logistic-regression :only (sigmoid)]))

(defn- add-bias-to-activation [a]
  (matrix/join-along 1 (utils/create-matrix-with-value [(matrix/row-count a) 1] 1) a))

;Calculate next layer's activation and attach bias unit.
;The given activation must include the bias unit so it is a m*(n+1) matrix.
(defn- calc-next-activation [current-activation theta]
  (let [z (matrix/mmul current-activation (matrix/transpose theta))
        a (matrix/emap sigmoid z)]
    (add-bias-to-activation a)))

(defn- gen-activations [activation-vec theta]
  (let [current-activation (last activation-vec)]
    (conj activation-vec (calc-next-activation current-activation theta))))

;X is training data, a m*n matrix
;Return [a1 a2 a3 ... aL]
(defn calc-activation-seq [X theta-seq]
  (reduce
    gen-activations
    [(add-bias-to-activation X)]
    theta-seq))

;TODO: no need to store all activations in this case
(defn predict [x theta-seq]
  (last (calc-activation-seq x theta-seq)))