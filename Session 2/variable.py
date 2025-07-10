# Basic Data types
# String
name = "Elijah"

# Number 
age = 29       # Integer,  "int" 
height = 180.0     # Float 7.8       "long float"

is_student = True        #False  Boolean

Score = None        # None

# Output the type of the variable
print(type(name))

# Collection Data Types
# List      item can be changed.  
a = [Score, "Andre", age]
a[1] = 200
print(a)

# Tuple     fixed value 
b = (age, name, 6896, "Andre", True)
#b[2] = height
#print(b)

# Dictionary or Dict          key-value pair
info = {
    "name": "Elijah",
    "age": 29,
    "score": 100.0
}
print(info["score"])

# Arithmetric Operators 
a, b =10, 3
print(a)
# + - * /
# floor division   \\. disgard the remainder = 3. Rounding to the nearest 1
c = a // b

# Modulus.   keep the remainder
d = 13 % 5

# Exponentiation (3 to the power of 2 2^3)
e = a ** b
print(e)

# > greater < smaller  == Equal
# != is not equal >= greater than equal <= smaller than equal

# and (times)
# not (if A is false, result is true ) input 1 - output 0 
# or (plus)
x = True
y = False

z = x or y
print(z) # and  Will be false
         # or  - if one of them is true, then the other one is true 

