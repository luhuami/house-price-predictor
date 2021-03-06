(ns neural.networks.iteration.forward-propagation
  (:require [clojure.core.matrix :as matrix])
  (:use [logistic-regression :only (sigmoid)]))

;a must be a n*1 activation matrix
(defn- add-bias-to-activation [a]
  (matrix/join-along 0 [[1]] a))

;Calculate next layer's activation, attach bias unit and save it to a vector.
;Given activations must include the bias unit and is a (n+1)*1 matrix.
(defn- calc-next-activation [activation-vec theta]
  (let [z (matrix/mmul theta (last activation-vec))
        a (matrix/emap sigmoid z)]
    (conj activation-vec (add-bias-to-activation a))))

;x must be a vector. One row of X.
;Return [a1 a2 a3 ... al]
(defn calc-activation-seq [x theta-seq]
  (reduce
    calc-next-activation
    [(add-bias-to-activation (matrix/transpose [x]))]
    theta-seq))

;TODO: no need to store all activations in this case
(defn calc [x theta-seq]
  (last (calc-activation-seq x theta-seq)))