from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
password = "1234"  # The password you're trying to use
hashed_password = pwd_context.hash(password)
print(hashed_password)