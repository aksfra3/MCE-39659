using Statistics, LinearAlgebra


K = 5

function knn_predict(X_train, y_train, input_point, k)

    distances = [norm(X_train[i, :] - input_point) for i in 1:size(X_train, 1)]
    
    nearest_indices = sortperm(distances)[1:k]
    
    return mean(y_train[nearest_indices])
end

println("Calculating KNN predictions...")
y_hat_knn = [knn_predict(X, y, X[i, :], K) for i in 1:size(X, 1)]

res_knn = y - y_hat_knn
knn_tracking_error = std(res_knn)
knn_correlation = cor(diff(y), diff(y_hat_knn))

println("--- KNN Performance (K=$K) ---")
println("Tracking Error: ", round(knn_tracking_error, digits=4))
println("Correlation:    ", round(knn_correlation, digits=4))

plot(df.Date, y, label="Actual TTF", color=:blue, alpha=0.6)
plot!(df.Date, y_hat_knn, label="KNN Prediction", color=:green, lw=1)
title!("TTF Replication: K-Nearest Neighbors")