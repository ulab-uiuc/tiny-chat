import logging
from pathlib import Path

import click
import uvicorn

from tiny_chat.config import ConfigManager

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--config', '-c', type=click.Path(exists=True), help='Configuration file path'
)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--workers', default=None, type=int, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(
    config: str | None,
    verbose: bool,
    host: str | None,
    port: int | None,
    workers: int | None,
    reload: bool,
) -> None:
    """Start the TinyChat API server

    This is the main entry point for running TinyChat as a web service.
    Other operations like health checks, configuration validation, and demos
    are available through the HTTP API or standalone Python scripts.
    """
    if config:
        config_manager = ConfigManager(Path(config))
    else:
        config_manager = ConfigManager()

    server_config = config_manager.load_config()

    # Setup logging
    log_config = server_config.logging.dict()
    if verbose:
        log_config['level'] = 'DEBUG'

    logger.info('TinyChat Server starting...')

    # Override config with CLI options
    api_config = server_config.api
    host = host or api_config.host
    port = port or api_config.port
    workers = workers or api_config.workers
    reload = reload or api_config.reload

    logger.info(f'Starting TinyChat API server on {host}:{port}')

    # Display available endpoints
    logger.info('Available endpoints:')
    logger.info('  GET  /health          - Health check')
    logger.info('  GET  /models          - List available models')
    logger.info('  POST /conversations   - Start conversation')
    logger.info('  GET  /conversations/{id} - Get conversation status')

    # Start API server
    uvicorn.run(
        'tiny_chat.server.api:app',
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level='info',
    )


if __name__ == '__main__':
    serve()
