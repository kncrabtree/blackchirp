#include "overlaymanagerwidget.h"
#include "bcexpoverlaydialog.h"
#include "overlaycheckboxdelegate.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDebug>
#include <QTableView>
#include <QHeaderView>
#include <QMessageBox>
#include <QProgressBar>
#include <gui/widget/ftmwviewwidget.h>

OverlayManagerWidget::OverlayManagerWidget(QWidget *parent, int number, const QVector<std::shared_ptr<OverlayBase>> &overlays)
    : QWidget{parent, Qt::Window}, p_plotIdDelegate(nullptr), p_numericDelegate(nullptr)
{
    // Set window attributes
    if(number > 0)
        setWindowTitle(QString("Overlay Manager: Experiment %1").arg(number));
    else
        setWindowTitle("Overlay Manager: Main Window");
    setWindowIcon(QIcon(":/icons/peak-find.svg")); // Temporary icon
    setAttribute(Qt::WA_DeleteOnClose);
    resize(800, 400); // Increased width to accommodate all columns

    setupUI();
    createTabs();
    populateWithExistingOverlays(overlays);
    updateButtonStates();
    
    // Progress indicator starts hidden
    p_progressWidget->setVisible(false);
}

OverlayManagerWidget::~OverlayManagerWidget()
{
    // Wait for any pending writes to complete before destruction
    if (p_overlayStorage) {
        p_overlayStorage->waitForPendingWrites();
        // Explicitly disconnect from overlay storage to prevent segfaults
        disconnect(p_overlayStorage.get(), nullptr, this, nullptr);
    }
}

void OverlayManagerWidget::setupUI()
{
    // Create main layout
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // Create toolbar
    p_toolBar = new QToolBar(this);
    p_toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    // Create actions
    p_addAction = p_toolBar->addAction(QIcon(":/icons/add.png"), "Add Overlay");
    p_addAction->setToolTip("Add a new overlay to the current plot");

    p_removeAction = p_toolBar->addAction(QIcon(":/icons/remove.png"), "Remove Overlay");
    p_removeAction->setToolTip("Remove the selected overlay");

    // Add separator
    p_toolBar->addSeparator();

    // Add raise parent action
    p_raiseParentAction = p_toolBar->addAction(QIcon(":/icons/go-up.svg"), "Show Parent");
    p_raiseParentAction->setToolTip("Bring the parent window to front");

    // Add spacer
    auto spacer = new QWidget;
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    p_toolBar->addWidget(spacer);

    // Create tab widget
    p_tabWidget = new QTabWidget(this);

    // Create progress indicator widget
    createProgressWidget();
    
    // Add widgets to layout
    mainLayout->addWidget(p_toolBar);
    mainLayout->addWidget(p_tabWidget);
    mainLayout->addWidget(p_progressWidget);

    // Connect signals
    connect(p_addAction, &QAction::triggered, this, &OverlayManagerWidget::addOverlay);
    connect(p_removeAction, &QAction::triggered, this, &OverlayManagerWidget::removeOverlay);
    connect(p_raiseParentAction, &QAction::triggered, this, &OverlayManagerWidget::raiseParent);
    connect(p_tabWidget, &QTabWidget::currentChanged, this, &OverlayManagerWidget::onTabChanged);
}

void OverlayManagerWidget::createTabs()
{
    // Use Q_ENUM to get all overlay types
    auto metaEnum = QMetaEnum::fromType<OverlayBase::OverlayType>();

    for(int i = 0; i < metaEnum.keyCount(); ++i)
    {
        QString typeName = metaEnum.key(i);
        auto typeValue = static_cast<OverlayBase::OverlayType>(metaEnum.value(i));

        // Create friendly names for tabs
        QString tabName;
        QWidget* tabWidget = nullptr;

        switch(typeValue)
        {
        case OverlayBase::BCExperiment:
            tabName = "BC Experiments";
            tabWidget = createBCExperimentTab();
            break;
        case OverlayBase::SPCAT:
            tabName = "SPCAT Catalogs";
            tabWidget = createPlaceholderTab(tabName);
            break;
        case OverlayBase::GenericXY:
            tabName = "Generic Data";
            tabWidget = createPlaceholderTab(tabName);
            break;
        default:
            tabName = typeName; // Fallback to enum name
            tabWidget = createPlaceholderTab(tabName);
            break;
        }

        // Store the overlay type as tab data for future use
        using namespace BC::Property::Overlay;
        tabWidget->setProperty(overlayType.toLocal8Bit().constData(), static_cast<int>(typeValue));

        // Add tab to tab widget
        p_tabWidget->addTab(tabWidget, tabName);

    }
}

QWidget *OverlayManagerWidget::createBCExperimentTab()
{
    auto tabWidget = new QWidget;
    auto tabLayout = new QVBoxLayout(tabWidget);

    // Create model and table view
    p_bcExperimentModel = new BCExperimentOverlayModel(this);
    p_bcExperimentTableView = new QTableView(tabWidget);
    p_bcExperimentTableView->setModel(p_bcExperimentModel);
    
    // Connect model signals to track overlay changes
    connect(p_bcExperimentModel, &BCExperimentOverlayModel::dataChanged, 
            this, &OverlayManagerWidget::onModelDataChanged);
    connect(p_bcExperimentModel, &BCExperimentOverlayModel::rowsInserted, 
            this, [this]() { resizeColumnsToContents(p_bcExperimentModel, p_bcExperimentTableView); });
    connect(p_bcExperimentModel, &BCExperimentOverlayModel::rowsRemoved, 
            this, [this]() { resizeColumnsToContents(p_bcExperimentModel, p_bcExperimentTableView); });
    
    // Connect selection signals to update button states
    connect(p_bcExperimentTableView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &OverlayManagerWidget::onSelectionChanged);
    
    // Register this model-view pair for automatic column resizing
    d_modelViewMap[p_bcExperimentModel] = p_bcExperimentTableView;

    // Configure table view
    p_bcExperimentTableView->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_bcExperimentTableView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    p_bcExperimentTableView->setAlternatingRowColors(true);
    p_bcExperimentTableView->setSortingEnabled(false); // Disable for now

    // Configure headers
    auto horizontalHeader = p_bcExperimentTableView->horizontalHeader();
    horizontalHeader->setStretchLastSection(false);
    horizontalHeader->setSectionResizeMode(QHeaderView::Interactive);

    auto verticalHeader = p_bcExperimentTableView->verticalHeader();
    verticalHeader->setDefaultSectionSize(25);
    verticalHeader->setVisible(false);

    // Set up table view with delegates and column widths
    setupTableView();

    tabLayout->addWidget(p_bcExperimentTableView);

    return tabWidget;
}

QWidget *OverlayManagerWidget::createPlaceholderTab(const QString &tabName)
{
    auto tabWidget = new QWidget;
    auto tabLayout = new QVBoxLayout(tabWidget);

    // Add placeholder content
    auto placeholderLabel = new QLabel(QString("Overlays of type '%1' will be managed here.\n\nImplementation coming soon...").arg(tabName));
    placeholderLabel->setAlignment(Qt::AlignCenter);
    placeholderLabel->setStyleSheet("color: gray; font-style: italic;");
    placeholderLabel->setWordWrap(true);

    tabLayout->addWidget(placeholderLabel);

    return tabWidget;
}

void OverlayManagerWidget::onTabChanged(int index)
{
    Q_UNUSED(index)
    updateButtonStates();
}

void OverlayManagerWidget::updateButtonStates()
{
    // Check if we have pending writes
    bool hasPendingWrites = p_overlayStorage && p_overlayStorage->hasPendingWrites();
    
    // Add button is disabled when there are pending writes
    p_addAction->setEnabled(!hasPendingWrites);
    
    // Remove button is enabled only when rows are selected and no pending writes
    bool hasSelection = false;
    
    // Get current tab's table view and check for selection
    int currentIndex = p_tabWidget->currentIndex();
    if (currentIndex >= 0) {
        auto currentTabWidget = p_tabWidget->currentWidget();
        if (currentTabWidget) {
            // Check if this is the BCExperiment tab
            using namespace BC::Property::Overlay;
            auto type = static_cast<OverlayBase::OverlayType>(currentTabWidget->property(overlayType.toLocal8Bit().constData()).toInt());
            
            if (type == OverlayBase::BCExperiment && p_bcExperimentTableView) {
                hasSelection = p_bcExperimentTableView->selectionModel()->hasSelection();
            }
            // Future overlay types can be added here
        }
    }
    
    p_removeAction->setEnabled(hasSelection && !hasPendingWrites);
}

void OverlayManagerWidget::addOverlay()
{
    // Ensure we have overlay storage connection
    if (!p_overlayStorage) {
        qDebug() << "Warning: No overlay storage connected to OverlayManagerWidget";
        return;
    }

    // Get current tab's overlay type
    auto currentTabWidget = p_tabWidget->currentWidget();
    if(currentTabWidget == nullptr)
        return;

    using namespace BC::Property::Overlay;
    auto type = static_cast<OverlayBase::OverlayType>(currentTabWidget->property(overlayType.toLocal8Bit().constData()).toInt());

    // Create overlay and get associated model based on type
    std::shared_ptr<OverlayBase> overlay = nullptr;
    OverlayTableModel* model = nullptr;
    
    switch(type) {
    case OverlayBase::BCExperiment:
        {
            // Get the FtmwViewWidget parent for plot names
            FtmwViewWidget* ftmwParent = qobject_cast<FtmwViewWidget*>(parentWidget());
            if(!ftmwParent) {
                qDebug() << "Warning: OverlayManagerWidget parent is not FtmwViewWidget";
                return;
            }
            
            // Get plot names and xRange from parent and create the dialog
            QStringList plotNames = ftmwParent->getPlotNames();
            auto xRange = ftmwParent->getMainPlotFt().xRange();
            BCExpOverlayDialog dialog(plotNames, xRange.first, xRange.second, ftmwParent);
            if(dialog.exec() == QDialog::Accepted) {
                // Create the overlay
                overlay = dialog.createOverlay();
                model = p_bcExperimentModel;
            }
            break;
        }
    case OverlayBase::SPCAT:
        // TODO: Implement SPCAT overlay creation
        // overlay = createSPCATOverlay();
        // model = p_spcatModel;
        qDebug() << "SPCAT overlay creation not yet implemented";
        break;
    case OverlayBase::GenericXY:
        // TODO: Implement GenericXY overlay creation
        // overlay = createGenericXYOverlay();
        // model = p_genericXYModel;
        qDebug() << "GenericXY overlay creation not yet implemented";
        break;
    default:
        qDebug() << "Unknown overlay type";
        break;
    }
    
    // Add overlay to storage if created successfully
    if (overlay != nullptr && model != nullptr) {
        // Add directly to overlay storage - this initiates async write
        if (p_overlayStorage->addOverlay(overlay)) {
            // Add to local model for display
            model->addOverlay(overlay);
            
            // Update UI state to show any pending writes
            updateButtonStates();
            
            qDebug() << "Overlay created and added to storage successfully";
        } else {
            qDebug() << "Failed to add overlay to storage";
        }
    }
}

void OverlayManagerWidget::removeOverlay()
{
    // Get current tab and its overlay type
    int currentIndex = p_tabWidget->currentIndex();
    if (currentIndex < 0)
        return;
        
    auto currentTabWidget = p_tabWidget->currentWidget();
    if (!currentTabWidget)
        return;
        
    using namespace BC::Property::Overlay;
    auto type = static_cast<OverlayBase::OverlayType>(currentTabWidget->property(overlayType.toLocal8Bit().constData()).toInt());
    
    // Get the appropriate model and table view for this overlay type
    OverlayTableModel* model = nullptr;
    QTableView* tableView = nullptr;
    
    switch (type) {
    case OverlayBase::BCExperiment:
        model = p_bcExperimentModel;
        tableView = p_bcExperimentTableView;
        break;
    case OverlayBase::SPCAT:
        // model = p_spcatModel;
        // tableView = p_spcatTableView;
        qDebug() << "SPCAT overlay removal not yet implemented";
        return;
    case OverlayBase::GenericXY:
        // model = p_genericXYModel;
        // tableView = p_genericXYTableView;
        qDebug() << "GenericXY overlay removal not yet implemented";
        return;
    default:
        qDebug() << "Unknown overlay type";
        return;
    }
    
    if (!model || !tableView)
        return;
        
    auto selectionModel = tableView->selectionModel();
    if (!selectionModel->hasSelection())
        return;
        
    // Get selected rows
    QModelIndexList selectedRows = selectionModel->selectedRows();
    if (selectedRows.isEmpty())
        return;
        
    // Create confirmation message
    QString message;
    if (selectedRows.size() == 1) {
        auto overlay = model->getOverlay(selectedRows.first().row());
        if (overlay) {
            message = QString("Are you sure you want to remove the overlay \"%1\"?").arg(overlay->getLabel());
        } else {
            message = "Are you sure you want to remove the selected overlay?";
        }
    } else {
        message = QString("Are you sure you want to remove %1 selected overlays?").arg(selectedRows.size());
    }
    
    // Show confirmation dialog
    int result = QMessageBox::question(this, "Remove Overlay", message,
                                     QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    
    if (result != QMessageBox::Yes)
        return;
        
    // Ensure we have overlay storage connection
    if (!p_overlayStorage) {
        qDebug() << "Warning: No overlay storage connected to OverlayManagerWidget";
        return;
    }
    
    // Remove overlays in reverse order to maintain valid indices
    QVector<int> rows;
    for (const auto& index : selectedRows) {
        rows.append(index.row());
    }
    std::sort(rows.begin(), rows.end(), std::greater<int>());
    
    for (int row : rows) {
        auto overlay = model->getOverlay(row);
        if (overlay) {
            // Remove from overlay storage - this will emit signals that FtmwViewWidget listens to
            if (p_overlayStorage->removeOverlay(overlay->getLabel())) {
                // Remove from local model for display
                model->removeOverlay(row);
                
                qDebug() << "Overlay removed from storage successfully";
            } else {
                qDebug() << "Failed to remove overlay from storage";
            }
        }
    }
}

void OverlayManagerWidget::raiseParent()
{
    // Get parent widget and bring it to front
    QWidget* w = parentWidget();
    if(w)
    {
        w->activateWindow();
        w->raise();
        w->show();
    }
}

void OverlayManagerWidget::populateWithExistingOverlays(const QVector<std::shared_ptr<OverlayBase>> &overlays)
{
    // Add existing overlays to the appropriate models
    for(const auto& overlay : overlays)
    {
        if(overlay == nullptr)
            continue;
            
        switch(overlay->type())
        {
        case OverlayBase::BCExperiment:
            if(p_bcExperimentModel != nullptr)
                p_bcExperimentModel->addOverlay(overlay);
            break;
        case OverlayBase::SPCAT:
            // TODO: Add to SPCAT model when implemented
            break;
        case OverlayBase::GenericXY:
            // TODO: Add to GenericXY model when implemented
            break;
        default:
            break;
        }
    }
}

void OverlayManagerWidget::setupPlotIdDelegate()
{
    // Get plot names from parent FtmwViewWidget
    QStringList plotNames;
    FtmwViewWidget* ftmwParent = qobject_cast<FtmwViewWidget*>(parentWidget());
    if (ftmwParent) {
        plotNames = ftmwParent->getPlotNames();
    }
    
    // Create and set the delegate for the PlotId column
    p_plotIdDelegate = new PlotIdComboBoxDelegate(plotNames, this);
    p_bcExperimentTableView->setItemDelegateForColumn(1, p_plotIdDelegate); // PlotIdColumn = 1
}

void OverlayManagerWidget::setupTableView()
{
    // Set up delegates
    setupPlotIdDelegate();
    
    // Create numeric delegate for numeric columns
    p_numericDelegate = new OverlayNumericDelegate(this);
    p_bcExperimentTableView->setItemDelegateForColumn(2, p_numericDelegate); // YScaleColumn = 2
    p_bcExperimentTableView->setItemDelegateForColumn(3, p_numericDelegate); // YOffsetColumn = 3
    p_bcExperimentTableView->setItemDelegateForColumn(4, p_numericDelegate); // XOffsetColumn = 4
    p_bcExperimentTableView->setItemDelegateForColumn(6, p_numericDelegate); // MinFreqValueColumn = 6
    p_bcExperimentTableView->setItemDelegateForColumn(8, p_numericDelegate); // MaxFreqValueColumn = 8
    
    // Create checkbox delegate for frequency enabled columns
    p_checkBoxDelegate = new OverlayCheckBoxDelegate(this);
    p_bcExperimentTableView->setItemDelegateForColumn(5, p_checkBoxDelegate); // MinFreqEnabledColumn = 5
    p_bcExperimentTableView->setItemDelegateForColumn(7, p_checkBoxDelegate); // MaxFreqEnabledColumn = 7
    
    // Set up column resize behavior
    resizeColumnsToContents(p_bcExperimentModel, p_bcExperimentTableView);
}

void OverlayManagerWidget::resizeColumnsToContents(const OverlayTableModel* model, QTableView* tableView)
{
    if (!model || !tableView) {
        return;
    }
    
    auto horizontalHeader = tableView->horizontalHeader();
    int columnCount = model->columnCount();
    int sourceFileColumn = 9; // SourceFileColumn from OverlayTableModel
    
    // Resize all columns except the source file column to contents
    for (int i = 0; i < columnCount; ++i) {
        if (i != sourceFileColumn) {
            tableView->resizeColumnToContents(i);
            horizontalHeader->setSectionResizeMode(i, QHeaderView::Interactive);
        }
    }
    
    // Set the source file column to stretch to fill remaining space
    if (sourceFileColumn < columnCount) {
        horizontalHeader->setSectionResizeMode(sourceFileColumn, QHeaderView::Stretch);
    }
    
    // Ensure minimum widths for readability
    QFontMetrics fm(tableView->font());
    
    // Set minimum widths for numeric columns to accommodate 12-character numbers
    int minNumericWidth = fm.horizontalAdvance("123456.1234") + 5;
    QVector<int> numericColumns = {2, 3, 4, 6, 8}; // YScale, YOffset, XOffset, MinFreqValue, MaxFreqValue columns
    for (int i : numericColumns) {
        if (i < columnCount) {
            int currentWidth = tableView->columnWidth(i);
            if (currentWidth < minNumericWidth) {
                tableView->setColumnWidth(i, minNumericWidth);
            }
        }
    }
}

void OverlayManagerWidget::onModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
    // Get the model that emitted the signal
    const OverlayTableModel* model = qobject_cast<const OverlayTableModel*>(topLeft.model());
    if (!model) {
        return;
    }
    
    // Process each changed cell
    for (int row = topLeft.row(); row <= bottomRight.row(); ++row) {
        for (int col = topLeft.column(); col <= bottomRight.column(); ++col) {
            auto overlay = model->getOverlay(row);
            if (!overlay) {
                continue;
            }
            
            // Check which column was changed and emit appropriate signal
            switch (col) {
            case 0: // LabelColumn - doesn't affect plot display
            case 5: // SourceFileColumn - doesn't affect plot display
                // No signal needed
                break;
            case 1: // PlotIdColumn - requires plot migration
                emit overlayPlotChanged(overlay, overlay->getPlotId());
                break;
            default: // All other columns affect plot data (YScale, YOffset, XOffset, and future columns)
                emit overlayDataChanged(overlay);
                break;
            }
        }
    }
    
    // Resize columns to contents for this model's view
    auto it = d_modelViewMap.find(model);
    if (it != d_modelViewMap.end()) {
        resizeColumnsToContents(model, it->second);
    }
}

void OverlayManagerWidget::onSelectionChanged()
{
    // Update button states when selection changes
    updateButtonStates();
}

void OverlayManagerWidget::createProgressWidget()
{
    // Create progress widget container
    p_progressWidget = new QWidget(this);
    auto progressLayout = new QHBoxLayout(p_progressWidget);
    progressLayout->setContentsMargins(5, 5, 5, 5);
    progressLayout->setSpacing(10);
    
    // Create progress label
    p_progressLabel = new QLabel("Writing overlay data...", p_progressWidget);
    p_progressLabel->setStyleSheet("font-weight: bold; color: #0066CC;");
    
    // Create progress bar
    p_progressBar = new QProgressBar(p_progressWidget);
    p_progressBar->setRange(0, 0); // Indeterminate progress
    p_progressBar->setMaximumHeight(16);
    
    // Add widgets to layout
    progressLayout->addWidget(p_progressLabel);
    progressLayout->addWidget(p_progressBar);
    progressLayout->addStretch();
    
    // Set widget background
    p_progressWidget->setStyleSheet("background-color: #F0F8FF; border: 1px solid #CCCCCC; border-radius: 4px;");
}

void OverlayManagerWidget::updateProgressDisplay(int pendingCount)
{
    if (pendingCount > 0) {
        // Show progress indicator
        if (pendingCount == 1) {
            p_progressLabel->setText("Writing 1 overlay...");
        } else {
            p_progressLabel->setText(QString("Writing %1 overlays...").arg(pendingCount));
        }
        p_progressWidget->setVisible(true);
    } else {
        // Hide progress indicator
        p_progressWidget->setVisible(false);
    }
}

void OverlayManagerWidget::showErrorNotification(const QString& overlayLabel, const QString& error)
{
    QString title = "Overlay Write Failed";
    QString message = QString("Failed to write overlay data for '%1'\n\nError: %2\n\nThe overlay has been removed.")
                     .arg(overlayLabel, error);
    
    QMessageBox::warning(this, title, message);
}

void OverlayManagerWidget::connectToOverlayStorage(std::shared_ptr<OverlayStorage> storage)
{
    // Disconnect from previous storage if any
    if (p_overlayStorage) {
        disconnect(p_overlayStorage.get(), nullptr, this, nullptr);
    }
    
    // Store new storage reference
    p_overlayStorage = storage;
    
    // Connect to storage signals
    if (p_overlayStorage) {
        connect(p_overlayStorage.get(), &OverlayStorage::overlayWriteCompleted,
                this, &OverlayManagerWidget::onOverlayWriteCompleted);
        connect(p_overlayStorage.get(), &OverlayStorage::overlayWriteFailed,
                this, &OverlayManagerWidget::onOverlayWriteFailed);
        connect(p_overlayStorage.get(), &OverlayStorage::pendingWritesChanged,
                this, &OverlayManagerWidget::onPendingWritesChanged);
        
        // Initialize progress display with current pending count
        updateProgressDisplay(p_overlayStorage->pendingWriteCount());
    }
}

void OverlayManagerWidget::onOverlayWriteCompleted(std::shared_ptr<OverlayBase> overlay)
{
    Q_UNUSED(overlay)
    // Write completed successfully - no action needed
    // Progress display will be updated via onPendingWritesChanged
}

void OverlayManagerWidget::onOverlayWriteFailed(std::shared_ptr<OverlayBase> overlay, QString error)
{
    if (!overlay) {
        return;
    }
    
    // Show error notification
    showErrorNotification(overlay->getLabel(), error);
    
    // Remove failed overlay from our models
    OverlayTableModel* model = nullptr;
    switch (overlay->type()) {
    case OverlayBase::BCExperiment:
        model = p_bcExperimentModel;
        break;
    case OverlayBase::SPCAT:
        // TODO: Set model when SPCAT model is implemented
        break;
    case OverlayBase::GenericXY:
        // TODO: Set model when GenericXY model is implemented
        break;
    default:
        break;
    }
    
    // Find and remove the overlay from the model
    if (model) {
        auto overlays = model->getAllOverlays();
        for (int i = 0; i < overlays.size(); ++i) {
            if (overlays[i] == overlay) {
                model->removeOverlay(i);
                break;
            }
        }
    }
}

void OverlayManagerWidget::onPendingWritesChanged(int count)
{
    // Update progress display
    updateProgressDisplay(count);
    
    // Update button states to potentially disable/enable overlay creation
    updateButtonStates();
}
