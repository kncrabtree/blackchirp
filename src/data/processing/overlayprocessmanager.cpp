#include "overlayprocessmanager.h"
#include "overlayoperation.h"

#include <QDebug>
#include <QMutexLocker>
#include <QDateTime>
#include <QtConcurrent/QtConcurrent>

OverlayProcessManager::OverlayProcessManager(QObject* parent)
    : QObject(parent),
      d_currentOperation(nullptr)
{
    // Create progress update timer
    p_progressTimer = new QTimer(this);
    p_progressTimer->setInterval(PROGRESS_UPDATE_INTERVAL_MS);
    p_progressTimer->setSingleShot(false);
    connect(p_progressTimer, &QTimer::timeout, this, &OverlayProcessManager::onProgressTimeout);
    
}

OverlayProcessManager::~OverlayProcessManager()
{
    // Cancel all pending operations
    cancelAllOperations();
    
    // Wait for current operation to complete
    if (d_currentOperation && d_currentOperation->watcher) {
        d_currentOperation->watcher->waitForFinished();
    }
    
}

OverlayProcessManager& OverlayProcessManager::instance()
{
    static OverlayProcessManager instance;
    return instance;
}

QString OverlayProcessManager::queueOperation(std::shared_ptr<OverlayOperation> operation, Priority priority)
{
    if (!operation) {
        qWarning() << "Attempted to queue null operation";
        return QString();
    }
    
    QMutexLocker locker(&d_mutex);
    
    // Generate unique operation ID
    QString operationId = generateOperationId();
    
    // Create operation info
    auto operationInfo = std::make_shared<OperationInfo>(operationId, operation, priority);
    
    // Add to queue and tracking
    d_queuedOperations.enqueue(operationInfo);
    d_allOperations[operationId] = operationInfo;
    
    // Sort queue by priority
    sortQueue();
    
    // Update statistics
    d_totalOperations++;
    
    emit queueSizeChanged(d_queuedOperations.size());
        
    // Start processing if not already running
    QMetaObject::invokeMethod(this, &OverlayProcessManager::processQueue, Qt::QueuedConnection);
    
    return operationId;
}

bool OverlayProcessManager::cancelOperation(const QString& operationId)
{
    QMutexLocker locker(&d_mutex);
    
    auto it = d_allOperations.find(operationId);
    if (it == d_allOperations.end()) {
        qWarning() << "Attempted to cancel unknown operation:" << operationId;
        return false;
    }
    
    auto operationInfo = it->second;
    
    if (operationInfo->state == OperationState::Completed ||
        operationInfo->state == OperationState::Failed ||
        operationInfo->state == OperationState::Cancelled) {
        // Operation already finished
        return false;
    }
    
    if (operationInfo->state == OperationState::Queued) {
        // Remove from queue
        for (int i = 0; i < d_queuedOperations.size(); ++i) {
            if (d_queuedOperations.at(i)->id == operationId) {
                d_queuedOperations.removeAt(i);
                break;
            }
        }
        
        operationInfo->state = OperationState::Cancelled;
        d_cancelledOperations++;
        
        emit queueSizeChanged(d_queuedOperations.size());
        emit operationCancelled(operationId);
        
        return true;
    }
    
    if (operationInfo->state == OperationState::Running) {
        // Cancel running operation
        if (operationInfo->operation->canCancel()) {
            operationInfo->operation->cancel();
            operationInfo->state = OperationState::Cancelled;
            d_cancelledOperations++;
            
            // Cancel the future if possible
            if (operationInfo->watcher) {
                operationInfo->watcher->cancel();
            }
            
            emit operationCancelled(operationId);
            return true;
        } else {
            qDebug() << "Operation" << operationId << "cannot be cancelled";
            return false;
        }
    }
    
    return false;
}

void OverlayProcessManager::cancelAllOperations()
{
    QMutexLocker locker(&d_mutex);
    
    // Cancel queued operations
    while (!d_queuedOperations.isEmpty()) {
        auto operationInfo = d_queuedOperations.dequeue();
        operationInfo->state = OperationState::Cancelled;
        d_cancelledOperations++;
        emit operationCancelled(operationInfo->id);
    }
    
    // Cancel current operation
    if (d_currentOperation && d_currentOperation->state == OperationState::Running) {
        if (d_currentOperation->operation->canCancel()) {
            d_currentOperation->operation->cancel();
            d_currentOperation->state = OperationState::Cancelled;
            d_cancelledOperations++;
            
            if (d_currentOperation->watcher) {
                d_currentOperation->watcher->cancel();
            }
            
            emit operationCancelled(d_currentOperation->id);
        }
    }
    
    emit queueSizeChanged(0);
    emit processingStateChanged(false);
    
}

bool OverlayProcessManager::isProcessing() const
{
    QMutexLocker locker(&d_mutex);
    return d_currentOperation && d_currentOperation->state == OperationState::Running;
}

bool OverlayProcessManager::hasQueuedOperations() const
{
    QMutexLocker locker(&d_mutex);
    return !d_queuedOperations.isEmpty();
}

int OverlayProcessManager::queueSize() const
{
    QMutexLocker locker(&d_mutex);
    return d_queuedOperations.size();
}

OverlayProcessManager::OperationState OverlayProcessManager::getOperationState(const QString& operationId) const
{
    QMutexLocker locker(&d_mutex);
    
    auto it = d_allOperations.find(operationId);
    if (it == d_allOperations.end()) {
        return OperationState::Failed; // Unknown operation
    }
    
    return it->second->state;
}

QString OverlayProcessManager::getOperationError(const QString& operationId) const
{
    QMutexLocker locker(&d_mutex);
    
    auto it = d_allOperations.find(operationId);
    if (it == d_allOperations.end()) {
        return "Unknown operation";
    }
    
    return it->second->errorMessage;
}

int OverlayProcessManager::getOperationProgress(const QString& operationId) const
{
    QMutexLocker locker(&d_mutex);
    
    auto it = d_allOperations.find(operationId);
    if (it == d_allOperations.end()) {
        return 0;
    }
    
    return it->second->progress.load();
}

QString OverlayProcessManager::getOperationMessage(const QString& operationId) const
{
    QMutexLocker locker(&d_mutex);
    
    auto it = d_allOperations.find(operationId);
    if (it == d_allOperations.end()) {
        return QString();
    }
    
    return it->second->progressMessage;
}

void OverlayProcessManager::processQueue()
{
    QMutexLocker locker(&d_mutex);
    
    // Don't start new operation if one is already running
    if (d_currentOperation && d_currentOperation->state == OperationState::Running) {
        return;
    }
    
    // Check if there are queued operations
    if (d_queuedOperations.isEmpty()) {
        if (d_currentOperation) {
            d_currentOperation.reset();
        }
        emit processingStateChanged(false);
        p_progressTimer->stop();
        return;
    }
    
    // Get next operation from queue
    d_currentOperation = d_queuedOperations.dequeue();
    emit queueSizeChanged(d_queuedOperations.size());
    
    // Update operation state
    d_currentOperation->state = OperationState::Running;
    d_currentOperation->startTime = QDateTime::currentDateTime();
    
    // Create watcher for this operation
    d_currentOperation->watcher = new QFutureWatcher<std::shared_ptr<OverlayBase>>(this);
    connect(d_currentOperation->watcher, &QFutureWatcher<std::shared_ptr<OverlayBase>>::finished,
            this, &OverlayProcessManager::onOperationFinished);
    
    // Start progress timer
    p_progressTimer->start();
    
    emit processingStateChanged(true);
    emit operationStarted(d_currentOperation->id);
        
    // Start the operation in background thread
    auto operation = d_currentOperation->operation;
    auto operationId = d_currentOperation->id;
    
    // Connect operation progress to our progress tracking
    connect(operation.get(), &OverlayOperation::progressChanged,
            this, [this, operationId](int progress, const QString& message) {
                updateOperationProgress(operationId, progress, message);
            });
    
    auto future = QtConcurrent::run([operation]() -> std::shared_ptr<OverlayBase> {
        return operation->execute(); // Let exceptions propagate to onOperationFinished
    });
    
    d_currentOperation->watcher->setFuture(future);
}

void OverlayProcessManager::onOperationFinished()
{
    QMutexLocker locker(&d_mutex);
    
    if (!d_currentOperation || !d_currentOperation->watcher) {
        qWarning() << "Operation finished but no current operation tracked";
        return;
    }
    
    auto watcher = d_currentOperation->watcher;
    QString operationId = d_currentOperation->id;
    
    if (watcher->isCanceled()) {
        d_currentOperation->state = OperationState::Cancelled;
        d_cancelledOperations++;
        emit operationCancelled(operationId);
    } else {
        try {
            auto result = watcher->result();
            if (result) {
                d_currentOperation->state = OperationState::Completed;
                d_completedOperations++;
                emit operationCompleted(operationId, result);
            } else {
                d_currentOperation->state = OperationState::Failed;
                // Check if the operation provided an error message in its progress
                QString progressMsg = d_currentOperation->progressMessage;
                if (progressMsg.startsWith("Error:")) {
                    d_currentOperation->errorMessage = progressMsg.mid(7).trimmed(); // Remove "Error: " prefix
                } else {
                    d_currentOperation->errorMessage = "Operation returned null result";
                }
                d_failedOperations++;
                emit operationFailed(operationId, d_currentOperation->errorMessage);
                qDebug() << "Operation" << operationId << "failed:" << d_currentOperation->errorMessage;
            }
        } catch (const std::exception& e) {
            d_currentOperation->state = OperationState::Failed;
            d_currentOperation->errorMessage = e.what();
            d_failedOperations++;
            emit operationFailed(operationId, d_currentOperation->errorMessage);
            qDebug() << "Operation" << operationId << "failed:" << e.what();
        }
    }
    
    // Cleanup watcher
    watcher->deleteLater();
    d_currentOperation->watcher = nullptr;
    
    // Clean up old completed operations
    cleanupCompletedOperations();
    
    // Process next operation
    QMetaObject::invokeMethod(this, &OverlayProcessManager::processQueue, Qt::QueuedConnection);
}

void OverlayProcessManager::onProgressTimeout()
{
    if (d_currentOperation && d_currentOperation->state == OperationState::Running) {
        int progress = d_currentOperation->progress.load();
        QString message = d_currentOperation->progressMessage;
        emit operationProgress(d_currentOperation->id, progress, message);
    }
}

void OverlayProcessManager::sortQueue()
{
    // Sort by priority (higher first), then by queue time (earlier first)
    std::sort(d_queuedOperations.begin(), d_queuedOperations.end(),
              [](const std::shared_ptr<OperationInfo>& a, const std::shared_ptr<OperationInfo>& b) {
                  if (a->priority != b->priority) {
                      return static_cast<int>(a->priority) > static_cast<int>(b->priority);
                  }
                  return a->queueTime < b->queueTime;
              });
}

QString OverlayProcessManager::generateOperationId()
{
    return QString("overlay_op_%1_%2")
           .arg(QDateTime::currentMSecsSinceEpoch())
           .arg(d_nextOperationId.fetch_add(1));
}

void OverlayProcessManager::cleanupCompletedOperations()
{
    // Remove old completed operations to prevent memory growth
    if (d_allOperations.size() <= MAX_COMPLETED_OPERATIONS) {
        return;
    }
    
    // Keep only the most recent completed operations
    QVector<QString> toRemove;
    QVector<std::pair<QDateTime, QString>> completedOps;
    
    for (const auto& [id, info] : d_allOperations) {
        if (info->state == OperationState::Completed ||
            info->state == OperationState::Failed ||
            info->state == OperationState::Cancelled) {
            completedOps.append({info->startTime, id});
        }
    }
    
    if (completedOps.size() > MAX_COMPLETED_OPERATIONS) {
        // Sort by completion time and remove oldest
        std::sort(completedOps.begin(), completedOps.end());
        
        int toRemoveCount = completedOps.size() - MAX_COMPLETED_OPERATIONS;
        for (int i = 0; i < toRemoveCount; ++i) {
            toRemove.append(completedOps[i].second);
        }
    }
    
    for (const QString& id : toRemove) {
        d_allOperations.erase(id);
    }
}

void OverlayProcessManager::updateOperationProgress(const QString& operationId, int progress, const QString& message)
{
    QMutexLocker locker(&d_mutex);
    
    auto it = d_allOperations.find(operationId);
    if (it != d_allOperations.end()) {
        it->second->progress.store(qBound(0, progress, 100));
        it->second->progressMessage = message;
    }
}
