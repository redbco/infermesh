//! HTTP and gRPC server implementations

use crate::config::RouterConfig;
use crate::handler::{RequestHandler, RequestContext};
use crate::router::RouterStats;
use crate::{Result, RouterError};

use axum::{
    extract::{Path, Query, State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use axum::serve;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    compression::CompressionLayer,
    trace::TraceLayer,
};
use tokio_stream::wrappers::TcpListenerStream;
use tracing::{info, error, debug};

/// HTTP server for handling REST API and WebSocket connections
#[derive(Clone)]
pub struct HttpServer {
    config: RouterConfig,
    handler: Arc<RequestHandler>,
    stats: Arc<RouterStats>,
}

/// gRPC server for handling gRPC requests
#[derive(Clone)]
pub struct GrpcServer {
    config: RouterConfig,
    handler: Arc<RequestHandler>,
    stats: Arc<RouterStats>,
}

/// Shared application state
#[derive(Clone)]
struct AppState {
    handler: Arc<RequestHandler>,
    stats: Arc<RouterStats>,
    config: RouterConfig,
}

impl HttpServer {
    /// Create a new HTTP server
    pub fn new(
        config: RouterConfig,
        handler: Arc<RequestHandler>,
        stats: Arc<RouterStats>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            handler,
            stats,
        })
    }

    /// Serve HTTP requests
    pub async fn serve(&self, bind_addr: &str) -> Result<()> {
        let addr: SocketAddr = bind_addr.parse()
            .map_err(|e| RouterError::Configuration(format!("Invalid bind address: {}", e)))?;

        info!("Starting HTTP server on {}", addr);

        let app_state = AppState {
            handler: self.handler.clone(),
            stats: self.stats.clone(),
            config: self.config.clone(),
        };

        let app = self.create_router(app_state);

        let listener = tokio::net::TcpListener::bind(&addr).await
            .map_err(|e| RouterError::Server(format!("Failed to bind to {}: {}", addr, e)))?;

        if let Err(e) = serve(listener, app).await {
            error!("HTTP server error: {}", e);
            return Err(RouterError::Server(format!("HTTP server failed: {}", e)));
        }

        Ok(())
    }

    /// Create the Axum router with all routes
    fn create_router(&self, state: AppState) -> AxumRouter {
        let mut router = AxumRouter::new()
            // Health check endpoint
            .route("/health", get(health_check))
            .route("/health/ready", get(readiness_check))
            .route("/health/live", get(liveness_check))
            
            // Metrics endpoint
            .route("/metrics", get(metrics_handler))
            
            // API endpoints
            .route("/v1/inference", post(inference_handler))
            .route("/v1/models", get(list_models))
            .route("/v1/models/:model_id", get(get_model))
            
            // WebSocket endpoint for streaming
            .route("/v1/stream", get(websocket_handler))
            
            // Router status and stats
            .route("/status", get(status_handler))
            .route("/stats", get(stats_handler))
            
            .with_state(state);

        // Add middleware layers
        let service = ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(CompressionLayer::new());

        router = router.layer(service);

        // Add CORS if enabled
        if self.config.enable_cors {
            router = router.layer(CorsLayer::permissive());
        }

        router
    }
}

impl GrpcServer {
    /// Create a new gRPC server
    pub fn new(
        config: RouterConfig,
        handler: Arc<RequestHandler>,
        stats: Arc<RouterStats>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            handler,
            stats,
        })
    }

    /// Serve gRPC requests
    pub async fn serve(&self, bind_addr: &str) -> Result<()> {
        let addr: SocketAddr = bind_addr.parse()
            .map_err(|e| RouterError::Configuration(format!("Invalid bind address: {}", e)))?;

        info!("Starting gRPC server on {}", addr);

        // For now, just create a basic gRPC server
        // In a full implementation, this would include the actual gRPC services
        let mut server_builder = tonic::transport::Server::builder();

        // Add reflection if enabled
        #[cfg(feature = "reflection")]
        if self.config.enable_grpc_reflection {
            let reflection_service = tonic_reflection::server::Builder::configure()
                .register_encoded_file_descriptor_set(include_bytes!("../../mesh-proto/src/descriptor.bin"))
                .build()
                .map_err(|e| RouterError::Server(format!("Failed to create reflection service: {}", e)))?;
            server_builder = server_builder.add_service(reflection_service);
        }

        // TODO: Add actual gRPC services here
        // server_builder = server_builder.add_service(SomeGrpcService::new(...));

        // For now, just log that gRPC server would start but don't actually start it
        // since we don't have any services to serve
        info!("gRPC server would start on {} (no services configured)", addr);
        
        // Return success for now
        // TODO: Uncomment when we have actual gRPC services to serve
        // if let Err(e) = server_builder.serve(addr).await {
        //     error!("gRPC server error: {}", e);
        //     return Err(RouterError::Server(format!("gRPC server failed: {}", e)));
        // }

        Ok(())
    }
}

// HTTP Handler functions

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> std::result::Result<impl IntoResponse, StatusCode> {
    state.stats.increment_http_requests();
    
    match state.handler.health_check().await {
        Ok(true) => Ok(Json(json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "uptime_seconds": state.stats.uptime_seconds()
        }))),
        Ok(false) => Err(StatusCode::SERVICE_UNAVAILABLE),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Readiness check endpoint
async fn readiness_check(State(state): State<AppState>) -> std::result::Result<impl IntoResponse, StatusCode> {
    state.stats.increment_http_requests();
    
    // Check if the router is ready to accept requests
    let ready = state.handler.health_check().await.unwrap_or(false);
    
    if ready {
        Ok(Json(json!({
            "status": "ready",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Liveness check endpoint
async fn liveness_check(State(state): State<AppState>) -> impl IntoResponse {
    state.stats.increment_http_requests();
    
    Json(json!({
        "status": "alive",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": state.stats.uptime_seconds()
    }))
}

/// Metrics endpoint (Prometheus format)
async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.stats.increment_http_requests();
    
    let metrics = format!(
        "# HELP mesh_router_requests_total Total number of requests\n\
         # TYPE mesh_router_requests_total counter\n\
         mesh_router_requests_total {}\n\
         # HELP mesh_router_responses_total Total number of responses\n\
         # TYPE mesh_router_responses_total counter\n\
         mesh_router_responses_total {}\n\
         # HELP mesh_router_errors_total Total number of errors\n\
         # TYPE mesh_router_errors_total counter\n\
         mesh_router_errors_total {}\n\
         # HELP mesh_router_http_requests_total Total number of HTTP requests\n\
         # TYPE mesh_router_http_requests_total counter\n\
         mesh_router_http_requests_total {}\n\
         # HELP mesh_router_grpc_requests_total Total number of gRPC requests\n\
         # TYPE mesh_router_grpc_requests_total counter\n\
         mesh_router_grpc_requests_total {}\n\
         # HELP mesh_router_active_connections Current number of active connections\n\
         # TYPE mesh_router_active_connections gauge\n\
         mesh_router_active_connections {}\n\
         # HELP mesh_router_uptime_seconds Router uptime in seconds\n\
         # TYPE mesh_router_uptime_seconds gauge\n\
         mesh_router_uptime_seconds {}\n",
        state.stats.total_requests(),
        state.stats.total_responses(),
        state.stats.total_errors(),
        state.stats.http_requests(),
        state.stats.grpc_requests(),
        state.stats.active_connections(),
        state.stats.uptime_seconds(),
    );
    
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        metrics,
    )
}

/// Inference endpoint
async fn inference_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> std::result::Result<impl IntoResponse, StatusCode> {
    state.stats.increment_http_requests();
    
    debug!("Received inference request: {:?}", payload);
    
    // Create request context
    let mut context = RequestContext::new();
    
    // Extract client information from headers
    if let Some(user_agent) = headers.get("user-agent") {
        if let Ok(ua) = user_agent.to_str() {
            context = context.with_user_agent(ua);
        }
    }
    
    // Extract model information from payload
    let model_name = payload.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    // Create labels for routing
    let labels = mesh_core::Labels::new(
        model_name,
        "latest",
        "default",
        "router"
    );
    
    // Route the request
    match state.handler.route_request(&context, &labels).await {
        Ok(target) => {
            info!("Routed request {} to {:?}", context.request_id, target.node_id);
            
            // TODO: Forward request to target and return response
            // For now, return a mock response
            state.stats.increment_responses();
            
            Ok(Json(json!({
                "request_id": context.request_id,
                "target_node": target.node_id.to_string(),
                "target_address": target.address.to_string(),
                "score": target.score,
                "response": "Mock inference response"
            })))
        }
        Err(e) => {
            error!("Failed to route request {}: {}", context.request_id, e);
            state.stats.increment_errors();
            Err(StatusCode::from_u16(e.to_status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        }
    }
}

/// List available models
async fn list_models(State(state): State<AppState>) -> std::result::Result<impl IntoResponse, StatusCode> {
    state.stats.increment_http_requests();
    
    // TODO: Query available models from state store
    let models = vec![
        json!({
            "id": "gpt-7b",
            "name": "GPT-7B",
            "status": "ready",
            "nodes": 2
        }),
        json!({
            "id": "llama-13b",
            "name": "LLaMA-13B",
            "status": "ready",
            "nodes": 1
        })
    ];
    
    state.stats.increment_responses();
    Ok(Json(json!({
        "models": models
    })))
}

/// Get specific model information
async fn get_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> std::result::Result<impl IntoResponse, StatusCode> {
    state.stats.increment_http_requests();
    
    debug!("Getting model info for: {}", model_id);
    
    // TODO: Query specific model from state store
    let model_info = json!({
        "id": model_id,
        "name": format!("Model {}", model_id),
        "status": "ready",
        "nodes": 1,
        "last_updated": chrono::Utc::now().to_rfc3339()
    });
    
    state.stats.increment_responses();
    Ok(Json(model_info))
}

/// WebSocket handler for streaming responses
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    state.stats.increment_websocket_connections();
    
    ws.on_upgrade(|socket| async move {
        debug!("WebSocket connection established");
        // TODO: Handle WebSocket communication
        // For now, just close the connection
        let _ = socket;
    })
}

/// Router status endpoint
async fn status_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.stats.increment_http_requests();
    
    Json(json!({
        "status": "running",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": state.stats.uptime_seconds(),
        "config": {
            "http_port": state.config.http_port,
            "grpc_port": state.config.grpc_port,
            "max_concurrent_requests": state.config.max_concurrent_requests,
            "enable_cors": state.config.enable_cors,
            "enable_websockets": state.config.enable_websockets
        }
    }))
}

/// Router statistics endpoint
async fn stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.stats.increment_http_requests();
    
    Json(json!({
        "requests_total": state.stats.total_requests(),
        "responses_total": state.stats.total_responses(),
        "errors_total": state.stats.total_errors(),
        "http_requests": state.stats.http_requests(),
        "grpc_requests": state.stats.grpc_requests(),
        "websocket_connections": state.stats.websocket_connections(),
        "active_connections": state.stats.active_connections(),
        "uptime_seconds": state.stats.uptime_seconds()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RouterConfigBuilder;
    use mesh_net::{ConnectionPool, TlsConfig};
    use mesh_state::{StateStore, QueryEngine, ScoringEngine};

    async fn create_test_app_state() -> AppState {
        let config = RouterConfigBuilder::new().build();
        let state_store = StateStore::new();
        let query_engine = QueryEngine::new(state_store.clone());
        let scoring_engine = ScoringEngine::new();
        let connection_pool = ConnectionPool::new(TlsConfig::insecure());
        
        let handler = Arc::new(RequestHandler::new(
            config.clone(),
            state_store,
            query_engine,
            scoring_engine,
            connection_pool,
        ).await.unwrap());
        
        let stats = Arc::new(RouterStats::default());
        
        AppState {
            handler,
            stats,
            config,
        }
    }

    #[tokio::test]
    async fn test_http_server_creation() {
        let config = RouterConfigBuilder::new().build();
        let state_store = StateStore::new();
        let query_engine = QueryEngine::new(state_store.clone());
        let scoring_engine = ScoringEngine::new();
        let connection_pool = ConnectionPool::new(TlsConfig::insecure());
        
        let handler = Arc::new(RequestHandler::new(
            config.clone(),
            state_store,
            query_engine,
            scoring_engine,
            connection_pool,
        ).await.unwrap());
        
        let stats = Arc::new(RouterStats::default());
        
        let server = HttpServer::new(config, handler, stats);
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_grpc_server_creation() {
        let config = RouterConfigBuilder::new().build();
        let state_store = StateStore::new();
        let query_engine = QueryEngine::new(state_store.clone());
        let scoring_engine = ScoringEngine::new();
        let connection_pool = ConnectionPool::new(TlsConfig::insecure());
        
        let handler = Arc::new(RequestHandler::new(
            config.clone(),
            state_store,
            query_engine,
            scoring_engine,
            connection_pool,
        ).await.unwrap());
        
        let stats = Arc::new(RouterStats::default());
        
        let server = GrpcServer::new(config, handler, stats);
        assert!(server.is_ok());
    }
}
