import 'package:flutter_test/flutter_test.dart';
import 'package:katago_onnx_mobile_example/main.dart';

void main() {
  testWidgets('Verify app starts', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that the app bar title is present
    expect(find.text('KataGo Mobile Example'), findsOneWidget);

    // Verify initial status
    expect(find.text('Status: Idle'), findsOneWidget);

    // Verify buttons exist
    expect(find.text('Start Native'), findsOneWidget);
    expect(find.text('Start ONNX'), findsOneWidget);
  });
}
