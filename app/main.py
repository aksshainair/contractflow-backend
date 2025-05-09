from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from datetime import timedelta
import uuid
from typing import List
import os
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

from .models import User, Document, DocumentStatus, DocumentUpdate
from .database import (
    get_user_by_email, create_user, create_document,
    get_document_by_id, update_document, get_documents_for_user,
    get_user_by_id
)
from .auth import (
    verify_password, get_password_hash, create_access_token,
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from . import clauses
from . import ai_chat

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://contractflow-frontend-.*\.vercel\.app",  # Allow all requests from vercel frontends
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(clauses.router, prefix="/api", tags=["clauses"])
app.include_router(ai_chat.router, prefix="/api", tags=["ai_chat"])

class EmailRequest(BaseModel):
    document_id: str
    recipient_email: str
    subject: str
    message: str

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": str(user["_id"]),
            "email": user["email"],
            "role": user["role"]
        }, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/")
async def create_new_user(user: User):
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["password"] = hashed_password
    user_dict["_id"] = str(uuid.uuid4())
    create_user(user_dict)
    return {"message": "User created successfully"}

@app.get("/documents/")
async def get_my_documents(
    status: DocumentStatus = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all documents assigned to current user, optionally filtered by status"""
    documents = get_documents_for_user(str(current_user["_id"]), current_user["role"])
    
    # Convert binary content to base64 for each document
    for doc in documents:
        if doc.get("content"):
            try:
                # If content is bytes, encode to base64
                if isinstance(doc["content"], bytes):
                    doc["content"] = base64.b64encode(doc["content"]).decode('utf-8')
            except Exception as e:
                print(f"Error encoding document content: {e}")
                doc["content"] = None
    
    if status:
        documents = [doc for doc in documents if doc["status"] == status]
    return documents

@app.post("/documents/{document_id}/approvers")
async def add_approvers(
    document_id: str,
    approver_ids: List[str],
    current_user: dict = Depends(get_current_user)
):
    document = get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if current_user["role"] != "reviewer" or str(current_user["_id"]) != document["reviewer_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the assigned reviewer can add approvers"
        )
    
    # Verify all approver IDs exist and are actually approvers
    for approver_id in approver_ids:
        approver = get_user_by_id(approver_id)
        if not approver or approver["role"] != "approver":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid approver ID: {approver_id}"
            )
    
    update_document(document_id, {"approvers": approver_ids})
    return {"message": "Approvers added successfully"}

@app.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    document = get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if (str(current_user["_id"]) != document["reviewer_id"] and 
        str(current_user["_id"]) not in document["approvers"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this document")
    
    # Convert binary content to base64
    if document.get("content"):
        try:
            if isinstance(document["content"], bytes):
                document["content"] = base64.b64encode(document["content"]).decode('utf-8')
        except Exception as e:
            print(f"Error encoding document content: {e}")
            document["content"] = None
    
    # Update status to IN_PROGRESS if it's NEW or PENDING
    if document["status"] in [DocumentStatus.NEW, DocumentStatus.PENDING]:
        update_document(document_id, {"status": DocumentStatus.IN_PROGRESS})
    
    return document

@app.put("/documents/{document_id}")
async def update_document_status(
    document_id: str,
    update: DocumentUpdate,
    current_user: dict = Depends(get_current_user)
):
    document = get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if current_user["role"] == "reviewer":
        if str(current_user["_id"]) != document["reviewer_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to update this document")
        
        # If reviewer is marking changes as complete
        if update.status == DocumentStatus.CHANGES_MADE:
            update_data = {
                "status": DocumentStatus.CHANGES_MADE,
                "changes_summary": update.changes_summary,
                "notes": update.notes
            }
            # Notify approvers (in a real system, this would send emails/notifications)
            for approver_id in document["approvers"]:
                approver = get_user_by_id(approver_id)
                print(f"Notifying approver {approver['email']} that changes are ready for review")
        else:
            update_data = update.dict(exclude_unset=True)
    else:  # approver
        if str(current_user["_id"]) not in document["approvers"]:
            raise HTTPException(status_code=403, detail="Not authorized to update this document")
        
        if update.status == DocumentStatus.APPROVED:
            # Mock email sending
            print(f"Document {document_id} approved and sent via email")
            update_data = {"status": update.status, "notes": update.notes}
        else:
            # Send back to reviewer with notes
            update_data = {
                "status": DocumentStatus.IN_PROGRESS,
                "notes": update.notes,
                "last_reviewed_by": str(current_user["_id"])
            }
            # Include content if it's being updated
            if update.content:
                update_data["content"] = update.content
    
    # Ensure status is always included in update
    if "status" not in update_data:
        update_data["status"] = document["status"]
    
    update_document(document_id, update_data)
    return {"message": "Document updated successfully"}

@app.get("/users/email/{email}")
async def get_user_by_email_endpoint(email: str):
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/api/documents/send-email")
async def send_email(request: EmailRequest):
    try:
        # Get document from database
        document = await get_document_by_id(request.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Create message
        msg = MIMEMultipart()
        msg['From'] = os.getenv('SMTP_EMAIL', 'aksshainair.work@gmail.com')
        msg['To'] = request.recipient_email
        msg['Subject'] = request.subject

        # Add message body
        msg.attach(MIMEText(request.message, 'plain'))

        # Add document as attachment
        if document.get('content'):
            attachment = MIMEApplication(document['content'].encode())
            attachment.add_header(
                'Content-Disposition',
                'attachment',
                filename=f"{document['title']}.docx"
            )
            msg.attach(attachment)

        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(
                os.getenv('SMTP_EMAIL', 'your-email@gmail.com'),
                os.getenv('SMTP_PASSWORD', 'your-app-password')
            )
            smtp.send_message(msg)

        return {"message": "Email sent successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}")
async def get_user_by_id_endpoint(user_id: str):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
