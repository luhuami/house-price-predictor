(ns linear-regression.linear-regression
  (:require [clojure.core.matrix :as matrix]))

(defn- not-empty? [X y]
  (and
    (> (matrix/row-count X) 0)
    (> (matrix/column-count X) 0)
    (> (matrix/row-count y) 0)))

(defn- same-row-count? [X y]
  (= (matrix/row-count X) (matrix/row-count y)))

;note vector and matrix are different
(defn- valid? [X y]
  (and
    (matrix/matrix? X)
    (matrix/matrix? y)
    (not-empty? X y)
    (= 1 (matrix/column-count y))
    (same-row-count? X y)))

(defn- create-bias-column [m]
  (matrix/broadcast [1] [m 1]))

(defn- add-bias [X]
  (let [bias-column (create-bias-column (first (matrix/shape X)))]
    (matrix/join-along 1 bias-column X)))

; Descent-Matrix = X' * (X * Theta - y)
(defn- calc-descent-matrix [X y Theta]
  (matrix/mmul
    (matrix/transpose X)
    (matrix/ereduce (matrix/mmul X Theta) y)))

(defn- perform-one-step-gradient-decent [X y Theta alpha]
  (let [m (matrix/row-count X)
        descent (matrix/emul (/ alpha m) (calc-descent-matrix X y Theta))]
    (matrix/ereduce Theta descent)))

(defn- create-initial-theta [n]
  (matrix/broadcast [0] [(inc n) 1]))

(defn perform-batch-gradient-decent [training-set y alpha iter]
  (let [X (add-bias training-set)]
    (loop [i 0
           Theta (create-initial-theta (matrix/column-count X))]
      (if (= i iter)
        Theta
        (recur
          (inc i)
          (perform-one-step-gradient-decent X y Theta alpha))))))