#ifndef OVERLAYPROCESSMANAGER_H
#define OVERLAYPROCESSMANAGER_H

#include <QObject>
#include <QQueue>
#include <QFutureWatcher>
#include <QMutex>
#include <QTimer>
#include <QDateTime>
#include <memory>
#include <atomic>

#include <data/experiment/overlaybase.h>

// Forward declarations
class OverlayOperation;

/**
 * @brief Singleton manager for background overlay processing operations
 * 
 * This class manages a queue of expensive overlay operations (convolution, 
 * file loading, overlay creation) and executes them in background threads.
 * It provides progress reporting, cancellation support, and operation 
 * prioritization while ensuring thread safety.
 * 
 * Modeled after the worker pattern used in FtmwViewWidget for FT processing.
 */
class OverlayProcessManager : public QObject
{
    Q_OBJECT

public:
    // Operation status tracking
    enum class OperationState {
        Queued,      // Operation is waiting in queue
        Running,     // Operation is currently executing
        Completed,   // Operation completed successfully
        Failed,      // Operation failed with error
        Cancelled    // Operation was cancelled by user
    };
    
    // Operation priority levels
    enum class Priority {
        Low = 0,
        Normal = 1,
        High = 2,
        Critical = 3
    };

    // Singleton access
    static OverlayProcessManager& instance();
    
    // Disable copy and move
    OverlayProcessManager(const OverlayProcessManager&) = delete;
    OverlayProcessManager& operator=(const OverlayProcessManager&) = delete;
    OverlayProcessManager(OverlayProcessManager&&) = delete;
    OverlayProcessManager& operator=(OverlayProcessManager&&) = delete;

    // Operation management
    QString queueOperation(std::shared_ptr<OverlayOperation> operation, Priority priority = Priority::Normal);
    bool cancelOperation(const QString& operationId);
    void cancelAllOperations();
    
    // Status queries
    bool isProcessing() const;
    bool hasQueuedOperations() const;
    int queueSize() const;
    OperationState getOperationState(const QString& operationId) const;
    QString getOperationError(const QString& operationId) const;
    
    // Progress tracking
    int getOperationProgress(const QString& operationId) const;
    QString getOperationMessage(const QString& operationId) const;

signals:
    // Operation lifecycle signals
    void operationStarted(const QString& operationId);
    void operationProgress(const QString& operationId, int percentage, const QString& message);
    void operationCompleted(const QString& operationId, std::shared_ptr<OverlayBase> result);
    void operationFailed(const QString& operationId, const QString& error);
    void operationCancelled(const QString& operationId);
    
    // Queue status signals
    void queueSizeChanged(int size);
    void processingStateChanged(bool isProcessing);

private slots:
    void processQueue();
    void onOperationFinished();
    void onProgressTimeout();

private:
    OverlayProcessManager(QObject* parent = nullptr);
    ~OverlayProcessManager();
    
    // Internal operation tracking structure
    struct OperationInfo {
        QString id;
        std::shared_ptr<OverlayOperation> operation;
        Priority priority;
        OperationState state;
        QString errorMessage;
        std::atomic<int> progress{0};
        QString progressMessage;
        QFutureWatcher<std::shared_ptr<OverlayBase>>* watcher;
        QDateTime queueTime;
        QDateTime startTime;
        
        OperationInfo(const QString& opId, std::shared_ptr<OverlayOperation> op, Priority prio)
            : id(opId), operation(op), priority(prio), state(OperationState::Queued),
              watcher(nullptr), queueTime(QDateTime::currentDateTime()) {}
    };
    
    // Queue management
    void sortQueue(); // Sort by priority and queue time
    QString generateOperationId();
    void cleanupCompletedOperations();
    void updateOperationProgress(const QString& operationId, int progress, const QString& message);
    
    // Thread safety
    mutable QMutex d_mutex;
    
    // Operation storage
    QQueue<std::shared_ptr<OperationInfo>> d_queuedOperations;
    std::map<QString, std::shared_ptr<OperationInfo>, std::less<>> d_allOperations; // All operations (queued, running, completed)
    std::shared_ptr<OperationInfo> d_currentOperation;
    
    // Configuration
    static const int MAX_COMPLETED_OPERATIONS = 100; // Keep history for debugging
    static const int PROGRESS_UPDATE_INTERVAL_MS = 100; // Progress update frequency
    
    // Progress tracking
    QTimer* p_progressTimer;
    
    // Statistics
    std::atomic<quint64> d_nextOperationId{1};
    std::atomic<int> d_totalOperations{0};
    std::atomic<int> d_completedOperations{0};
    std::atomic<int> d_failedOperations{0};
    std::atomic<int> d_cancelledOperations{0};
};

#endif // OVERLAYPROCESSMANAGER_H
