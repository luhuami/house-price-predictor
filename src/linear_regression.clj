(ns linear-regression
  (:require [clojure.core.matrix :as matrix]))

(defn- create-bias-column [m]
  (matrix/broadcast [1] [m 1]))

(defn- add-bias [X]
  (let [bias-column (create-bias-column (matrix/row-count X))]
    (matrix/join-along 1 bias-column X)))

;Descent-Matrix = X' * (X * Theta - y)
(defn- calc-descent-matrix [X y Theta]
  (matrix/mmul
    (matrix/transpose X)
    (matrix/sub (matrix/mmul X Theta) y)))

(defn- perform-one-step-gradient-decent [X y Theta alpha]
  (let [m (matrix/row-count X)
        descent (matrix/mul (/ alpha m) (calc-descent-matrix X y Theta))]
    (matrix/sub Theta descent)))

(defn- create-initial-theta [n]
  (matrix/broadcast [0] [n 1]))

(defn perform-batch-gradient-decent [training-set y alpha iter]
  (let [X (add-bias training-set)]
    (loop [i 0
           Theta (create-initial-theta (matrix/column-count X))]
      (if (= i iter)
        Theta
        (recur
          (inc i)
          (perform-one-step-gradient-decent X y Theta alpha))))))