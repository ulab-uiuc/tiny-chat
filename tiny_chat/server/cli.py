import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn

from .config import ConfigManager, get_config
from .monitoring import get_metrics_collector, setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    '--config', '-c', type=click.Path(exists=True), help='Configuration file path'
)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(config: Optional[str], verbose: bool):
    """TinyChat Server CLI"""
    if config:
        config_manager = ConfigManager(Path(config))
    else:
        config_manager = ConfigManager()

    # Load configuration
    server_config = config_manager.load_config()

    # Setup logging
    log_config = server_config.logging.dict()
    if verbose:
        log_config['level'] = 'DEBUG'

    setup_logging(log_config)

    logger.info('TinyChat Server CLI initialized')


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--workers', default=None, type=int, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(
    host: Optional[str], port: Optional[int], workers: Optional[int], reload: bool
):
    """Start the TinyChat API server"""
    config = get_config()

    # Override config with CLI options
    api_config = config.api
    host = host or api_config.host
    port = port or api_config.port
    workers = workers or api_config.workers
    reload = reload or api_config.reload

    logger.info(f'Starting TinyChat API server on {host}:{port}')

    # Start metrics server if enabled
    if config.enable_metrics:
        try:
            metrics_collector = get_metrics_collector()
            metrics_collector.start_prometheus_server(config.metrics_port)
        except Exception as e:
            logger.warning(f'Failed to start metrics server: {e}')

    # Start API server
    uvicorn.run(
        'tiny_chat.server.api:app',
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level='info',
    )


@cli.command()
def check_config():
    """Validate configuration"""
    try:
        config = get_config()
        click.echo('Configuration is valid')
        click.echo(f'Models: {list(config.models.keys())}')
        click.echo(f'Default model: {config.default_model}')
        click.echo(f'Evaluators: {len(config.evaluators)}')

    except Exception as e:
        click.echo(f'Configuration error: {e}', err=True)
        sys.exit(1)


@cli.command()
async def check_health():
    """Check server health"""
    from .core import create_server

    try:
        async with create_server() as server:
            click.echo('Server initialization successful')

            # Check model health
            model_info = await server.get_model_info()

            healthy_models = sum(1 for info in model_info.values() if info['healthy'])
            total_models = len(model_info)

            click.echo(f'Models: {healthy_models}/{total_models} healthy')

            for name, info in model_info.items():
                status = '1' if info['healthy'] else '0'
                click.echo(f"  {status} {name} ({info['type']})")

    except Exception as e:
        click.echo(f'Health check failed: {e}', err=True)
        sys.exit(1)


@cli.command()
@click.option('--agents', '-a', multiple=True, help='Agent names')
@click.option('--model', '-m', help='Model to use')
@click.option('--turns', '-t', type=int, help='Maximum turns')
@click.option('--scenario', '-s', help='Scenario description')
def demo(
    agents: tuple, model: Optional[str], turns: Optional[int], scenario: Optional[str]
):
    """Run a demo conversation"""
    from .core import create_server

    async def run_demo():
        async with create_server() as server:
            # Default agents if none provided
            if not agents:
                agents_list = ['Alice', 'Bob']
            else:
                agents_list = list(agents)

            # Create agent configs
            agent_configs = [{'name': name, 'type': 'llm'} for name in agents_list]

            # Run conversation
            try:
                episode_log = await server.run_conversation(
                    agent_configs=agent_configs,
                    scenario=scenario or 'A casual conversation between friends',
                    model_name=model,
                    max_turns=turns,
                    return_log=True,
                )

                if episode_log:
                    click.echo('Demo conversation completed')
                    click.echo(f"Agents: {', '.join(episode_log.agents)}")
                    click.echo(f'Reasoning: {episode_log.reasoning}')
                else:
                    click.echo('Demo conversation completed (no log returned)')

            except Exception as e:
                click.echo(f'Demo failed: {e}', err=True)

    asyncio.run(run_demo())


@cli.command()
def metrics():
    """Show server metrics"""
    from .monitoring import get_server_health

    health = get_server_health()
    metrics = health['metrics']

    click.echo(f"Server Status: {health['status']}")
    click.echo(f"Total Conversations: {metrics['total_conversations']}")
    click.echo(f"Success Rate: {metrics['success_rate_percent']:.1f}%")
    click.echo(f"Active Conversations: {metrics['active_conversations']}")
    click.echo(f"Average Turns: {metrics['average_turns_per_conversation']:.1f}")
    click.echo(f"Average Duration: {metrics['average_duration_seconds']:.1f}s")

    if metrics['model_usage']:
        click.echo('Model Usage:')
        for model, count in metrics['model_usage'].items():
            click.echo(f'  {model}: {count}')


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export_config(output: Optional[str]):
    """Export current configuration to file"""
    config = get_config()
    config_manager = ConfigManager()

    if output:
        output_path = Path(output)
    else:
        output_path = Path('tiny_chat_config_export.yaml')

    config_manager.config_path = output_path
    config_manager.save_config(config)

    click.echo(f'Configuration exported to {output_path}')


if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    cli()
