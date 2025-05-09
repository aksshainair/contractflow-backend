from fastapi import FastAPI

def create_app():
    app = FastAPI()

    # Register routes here
    @app.get("/")
    async def index():
        return {"message": "Hello, World!"}

    return app
