import { FeatureGate } from './FeatureGate';

function TumorTracker({ darkMode }) {
  return (
    <FeatureGate feature="tumor_tracking" darkMode={darkMode}>
      <div>
        {/* Your tumor tracking UI */}
      </div>
    </FeatureGate>
  );
}

export default TumorTracker;
