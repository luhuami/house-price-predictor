(ns linear-regression.linear-regression
  (:require [clojure.core.matrix :as matrix]))

(defn- same-row-count? [X y]
  (= (matrix/row-count X) (matrix/row-count y)))

(defn- create-bias-column [m]
  (matrix/broadcast [1] [m 1]))

(defn- add-bias [X]
  (if (matrix/vec? X)
    (matrix/join [1] X)
    (let [bias-column (create-bias-column (first (matrix/shape X)))]
      (matrix/join-along 1 bias-column X))))

; Descent-Matrix = X' * (X * Theta - y)
(defn- calc-descent-matrix [X y Theta]
  (matrix/mmul
    (matrix/transpose X)
    (matrix/ereduce (matrix/mmul X Theta) y)))

(defn- perform-one-step-gradient-decent [X y Theta alpha]
  (let [m (matrix/row-count X)
        descent (matrix/emul (/ alpha m) (calc-descent-matrix X y Theta))]
    (matrix/ereduce Theta descent)))

(defn- create-initial-theta [X]
  (matrix/broadcast [0] [(matrix/column-count X) 1]))

(defn perform-batch-gradient-decent [training-set y Theta alpha iter]
  (loop [i 0
          X (add-bias training-set)
          Theta (create-initial-theta X)]
    (if (= i iter)
      Theta
      (recur
        (inc iter)
        X
        (perform-one-step-gradient-decent X y Theta alpha)))))