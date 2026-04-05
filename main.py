from fastapi import FastAPI

# Create FastAPI app instance
app = FastAPI(title="Terra Nova API")

# Define a simple GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/about")
def read_about():
    return {"message": "About Terra Nova API"}

@app.get("/name/{name}")
def read_name(name: str):
    return {"message": f"Hello, {name}!"}