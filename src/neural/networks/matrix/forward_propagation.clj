(ns neural.networks.matrix.forward-propagation
  (:require [clojure.core.matrix :as matrix]
            [utils.matrix :as utils])
  (:use [logistic-regression :only (sigmoid)]))

(defn- add-bias [a]
  (matrix/join-along
    1
    (utils/create-matrix-with-value [(matrix/row-count a) 1] 1) a))

;Calculate next layer's activation and attach bias unit.
;The given activation must include the bias unit so it is a m*(n+1) matrix.
(defn- calc-next-activation [current-activation theta]
  (let [z (matrix/mmul current-activation (matrix/transpose theta))
        a (matrix/emap sigmoid z)]
    (add-bias a)))

(defn- remove-bias-for-last-matrix [activations]
  (concat (drop-last activations)
          (list (utils/remove-first-column (last activations)))))

(defn calc-activation-seq [X theta-seq]
  (remove-bias-for-last-matrix (reductions
                                 calc-next-activation
                                 (add-bias X)
                                 theta-seq)))

(defn predict [x theta-seq]
  (last (calc-activation-seq x theta-seq)))