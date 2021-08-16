#ifndef EXPERIMENTSUMMARYWIDGET_H
#define EXPERIMENTSUMMARYWIDGET_H

#include <QWidget>
#include <QSortFilterProxyModel>

class Experiment;
class QLineEdit;
class QTreeView;

class ExperimentSummaryWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ExperimentSummaryWidget(QWidget *parent = nullptr);
    void setExperiment(Experiment *exp);

private:
    QLineEdit *p_le;
    QTreeView *p_view;

signals:


    // QWidget interface
public:
    QSize sizeHint() const override;
};

class ExptProxyModel : public QSortFilterProxyModel
{
    Q_OBJECT
public:
    ExptProxyModel(QObject *parent = nullptr) : QSortFilterProxyModel(parent) {}

    // QSortFilterProxyModel interface
protected:
    bool filterAcceptsRow(int source_row, const QModelIndex &source_parent) const override
    {
        QVector<QModelIndex> cols;
        auto n = sourceModel()->columnCount(source_parent);
        for(int i=0; i<n; ++i)
        {
            auto idx = sourceModel()->index(source_row,i,source_parent);
            if(!idx.isValid())
                return false;
            cols.append(idx);
        }

        auto rows = sourceModel()->rowCount(cols.constFirst());
        for(int i=0; i<rows; ++i)
        {
            if(filterAcceptsRow(i,cols.constFirst()))
                return true;
        }

        for(int i=0; i<cols.size(); ++i)
        {
            if(cols.at(i).data().toString().contains(filterRegularExpression()))
                return true;
        }

        return false;
    }
};

#endif // EXPERIMENTSUMMARYWIDGET_H
