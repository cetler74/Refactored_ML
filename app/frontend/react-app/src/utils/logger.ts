/**
 * Simple console logger utility with prefix
 */
export const logger = {
    log: (...args: any[]) => console.log('[App]', ...args),
    info: (...args: any[]) => console.info('[App]', ...args),
    warn: (...args: any[]) => console.warn('[App]', ...args),
    error: (...args: any[]) => console.error('[App]', ...args),
}; 