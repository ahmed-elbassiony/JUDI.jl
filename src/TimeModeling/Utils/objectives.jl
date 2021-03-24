# March 2021
# Author: Mathias Louboutin (mlouboutin3@gatech.edu)

l2(x, y) = (.5f0*norm(x - y)^2, x - y)
l1(x, y) = (norm(x-y, 1), sign.(x - y))
lp(x, y, p::Integer=2) = (norm(x - y, p)^p, p*(x - y).^(p-1))

function L2Misift(x, y, normalize::Bool=False, np::Integer=2, )
    nx, ny = norm(x, p), norm(y, p)
    f = .5f0*norm(x/nx - y/ny)^2
    g = 1/nx * (dot(x, y)/(nx^2*ny) * x - y / ny) 
    return f, g
end

function PhaseMisfit(x, y)
    nx, ny = norm(x, p), norm(y, p)
    f = nx - dot(x, y)/ny
    g = x/nx - y /ny
    return f, g
end
