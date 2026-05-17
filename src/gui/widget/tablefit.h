#ifndef TABLEFIT_H
#define TABLEFIT_H

#include <QTableWidget>
#include <QHeaderView>

// Make a settings-style QTableWidget report its true content height.
//
// QTableWidget::sizeHint() returns an arbitrary default (~256x192)
// regardless of how many rows it holds, so a table with two rows claims
// as much vertical space as one with twenty. The side-panel docks rely
// on an honest content height to size themselves, so cap the table at
// the summed height of its (visible) rows plus the frame, and let the
// vertical scrollbar appear only when the panel is squeezed shorter
// than that. Rows must already be QHeaderView::ResizeToContents.
//
// Idempotent: safe to call again after rows are shown/hidden so the
// cap tracks the new content height.
inline void fitTableToContents(QTableWidget *table)
{
    table->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    table->resizeRowsToContents();

    int h = table->verticalHeader()->length() + 2 * table->frameWidth();
    if (table->horizontalHeader()->isVisible())
        h += table->horizontalHeader()->height();

    table->setMaximumHeight(h);
    table->setSizePolicy(table->sizePolicy().horizontalPolicy(),
                         QSizePolicy::Maximum);
}

#endif // TABLEFIT_H
