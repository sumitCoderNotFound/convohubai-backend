"""
ConvoHubAI - Conversation Flow Model
Visual workflow builder for AI agents
"""
from sqlalchemy import Column, String, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class ConversationFlow(BaseModel):
    """Conversation flow/workflow for an agent."""
    __tablename__ = "conversation_flows"
    
    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Flow data (nodes and edges from React Flow)
    nodes = Column(JSON, default=[])  # List of nodes
    edges = Column(JSON, default=[])  # List of connections
    
    # Viewport (for saving zoom/pan position)
    viewport = Column(JSON, default={"x": 0, "y": 0, "zoom": 1})
    
    # Status
    is_active = Column(Boolean, default=False)
    is_draft = Column(Boolean, default=True)
    
    # Version control
    version = Column(String(20), default="1.0")
    
    # Agent relationship
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Workspace
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Relationships
    agent = relationship("Agent", foreign_keys=[agent_id])
    workspace = relationship("Workspace")
    
    def __repr__(self):
        return f"<ConversationFlow {self.name}>"


class FlowNode(BaseModel):
    """Individual node in a conversation flow (for detailed tracking)."""
    __tablename__ = "flow_nodes"
    
    # Node info
    node_id = Column(String(100), nullable=False)  # Frontend-generated ID
    node_type = Column(String(50), nullable=False)  # start, message, question, condition, action, end
    
    # Position
    position_x = Column(String(20), default="0")
    position_y = Column(String(20), default="0")
    
    # Node data
    label = Column(String(255), nullable=True)
    content = Column(Text, nullable=True)  # Message content or question text
    
    # For question nodes
    variable_name = Column(String(100), nullable=True)  # Store answer in this variable
    expected_type = Column(String(50), nullable=True)  # text, number, yes_no, choice
    choices = Column(JSON, nullable=True)  # For multiple choice questions
    
    # For condition nodes
    conditions = Column(JSON, nullable=True)  # List of conditions [{variable, operator, value, next_node}]
    
    # For action nodes
    action_type = Column(String(50), nullable=True)  # webhook, api_call, set_variable, transfer
    action_config = Column(JSON, nullable=True)  # Action-specific configuration
    
    # Flow reference
    flow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_flows.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Relationships
    flow = relationship("ConversationFlow", foreign_keys=[flow_id])
    
    def __repr__(self):
        return f"<FlowNode {self.node_type}: {self.label}>"